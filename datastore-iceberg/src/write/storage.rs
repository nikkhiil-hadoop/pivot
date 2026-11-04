//! The iceberg crate's file IO over a table's store.
//!
//! A commit is produced by the crate's own transaction, which writes the
//! manifest and manifest list of the snapshot it adds through the file IO of
//! the table the catalog loaded. The commit's catalog client builds that file
//! IO with the factory here, so the objects land through the table's store,
//! signed with the keys the catalog vended for the table, exactly as its data
//! files did. The store's calls block, so each runs off the runtime's workers.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use iceberg::io::{
    FileMetadata, FileRead, FileWrite, InputFile, OutputFile, Storage, StorageConfig,
    StorageFactory,
};
use iceberg::{Error as IcebergError, ErrorKind};
use object_storage::{ExternalStoreFactory, ObjectPath};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::store::{TableStore, read_vended_s3_credentials};
use crate::write::WriteTarget;

/// Builds the file IO of the table a commit's catalog client loads: over the
/// store the INSERT's target writes through, renewed with the keys the load
/// vended, so a commit signs with keys that have not expired under a long
/// write.
pub(crate) struct TableStorageFactory {
    target: Arc<WriteTarget>,
    store_factory: Arc<dyn ExternalStoreFactory>,
}

impl TableStorageFactory {
    pub(crate) fn new(
        target: Arc<WriteTarget>,
        store_factory: Arc<dyn ExternalStoreFactory>,
    ) -> Self {
        Self {
            target,
            store_factory,
        }
    }
}

#[typetag::serde]
impl StorageFactory for TableStorageFactory {
    fn build(&self, config: &StorageConfig) -> iceberg::Result<Arc<dyn Storage>> {
        let table = self.target.name.to_string();
        let vended =
            read_vended_s3_credentials(&table, config.props()).map_err(into_iceberg_error)?;
        // The store is opened on the directory the commit's objects go to:
        // any location under the table's root names the root.
        let metadata_location = self.target.store.uri_of(&self.target.metadata_dir);
        let store = TableStore::open_or_renew(
            Some(&self.target.store),
            &table,
            &metadata_location,
            self.store_factory.as_ref(),
            vended,
        )
        .map_err(into_iceberg_error)?;
        Ok(Arc::new(TableStorage { store }))
    }
}

impl fmt::Debug for TableStorageFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableStorageFactory")
            .field("table", &self.target.name.to_string())
            .finish_non_exhaustive()
    }
}

/// The crate's storage over one table's store: a file is located by its full
/// location, which must be under the table's root, and read or written whole.
#[derive(Clone)]
pub(crate) struct TableStorage {
    store: TableStore,
}

impl TableStorage {
    /// Run `call` against the store on a blocking thread: the store's calls
    /// block on the network, and the runtime's workers must not.
    async fn blocking<T: Send + 'static>(
        &self,
        call: impl FnOnce(&TableStore) -> crate::Result<T> + Send + 'static,
    ) -> iceberg::Result<T> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || call(&store))
            .await
            .map_err(|panicked| {
                IcebergError::new(ErrorKind::Unexpected, "a store call panicked")
                    .with_source(panicked)
            })?
            .map_err(into_iceberg_error)
    }

    /// The size of the object at `path`, or `None` when there is none.
    async fn size_of(&self, path: &str) -> iceberg::Result<Option<u64>> {
        let path = path.to_string();
        self.blocking(move |store| store.object_size(&path)).await
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for TableStorage {
    async fn exists(&self, path: &str) -> iceberg::Result<bool> {
        Ok(self.size_of(path).await?.is_some())
    }

    async fn metadata(&self, path: &str) -> iceberg::Result<FileMetadata> {
        let size = self.size_of(path).await?.ok_or_else(|| not_found(path))?;
        Ok(FileMetadata { size })
    }

    async fn read(&self, path: &str) -> iceberg::Result<Bytes> {
        let location = path.to_string();
        let bytes = self
            .blocking(move |store| Ok(store.store().get(&store.key_of(&location)?)?))
            .await?
            .ok_or_else(|| not_found(path))?;
        Ok(Bytes::from(bytes))
    }

    async fn reader(&self, path: &str) -> iceberg::Result<Box<dyn FileRead>> {
        Ok(Box::new(ObjectRead {
            bytes: self.read(path).await?,
        }))
    }

    async fn write(&self, path: &str, bytes: Bytes) -> iceberg::Result<()> {
        let location = path.to_string();
        self.blocking(move |store| Ok(store.store().put(&store.key_of(&location)?, &bytes)?))
            .await
    }

    async fn writer(&self, path: &str) -> iceberg::Result<Box<dyn FileWrite>> {
        Ok(Box::new(ObjectWrite {
            storage: self.clone(),
            path: path.to_string(),
            buffer: Vec::new(),
        }))
    }

    async fn delete(&self, path: &str) -> iceberg::Result<()> {
        let location = path.to_string();
        self.blocking(move |store| Ok(store.store().delete(&store.key_of(&location)?)?))
            .await
    }

    async fn delete_prefix(&self, path: &str) -> iceberg::Result<()> {
        let location = path.to_string();
        self.blocking(move |store| delete_under(store, &store.key_of(&location)?))
            .await
    }

    async fn delete_stream(&self, mut paths: BoxStream<'static, String>) -> iceberg::Result<()> {
        while let Some(path) = paths.next().await {
            self.delete(&path).await?;
        }
        Ok(())
    }

    fn new_input(&self, path: &str) -> iceberg::Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> iceberg::Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

impl fmt::Debug for TableStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableStorage")
            .field("table", &self.store.table_name())
            .finish_non_exhaustive()
    }
}

/// Delete every object under `prefix`, level by level.
fn delete_under(store: &TableStore, prefix: &ObjectPath) -> crate::Result<()> {
    let listing = store.store().list(prefix)?;
    for object in listing.objects {
        store
            .store()
            .delete(&prefix.join(object.file.path.as_str()))?;
    }
    for child in listing.prefixes {
        delete_under(store, &prefix.join(child.as_str()))?;
    }
    Ok(())
}

/// A file read whole when opened: what the crate reads this way, a manifest
/// list or a manifest, is small and read once.
struct ObjectRead {
    bytes: Bytes,
}

#[async_trait]
impl FileRead for ObjectRead {
    async fn read(&self, range: Range<u64>) -> iceberg::Result<Bytes> {
        let out_of_bounds = || {
            IcebergError::new(
                ErrorKind::DataInvalid,
                format!(
                    "range {}..{} is outside the object's {} bytes",
                    range.start,
                    range.end,
                    self.bytes.len()
                ),
            )
        };
        let start = usize::try_from(range.start).map_err(|_| out_of_bounds())?;
        let end = usize::try_from(range.end).map_err(|_| out_of_bounds())?;
        if start > end || end > self.bytes.len() {
            return Err(out_of_bounds());
        }
        Ok(self.bytes.slice(start..end))
    }
}

/// A file written whole when closed: an object store takes an object in one
/// put.
struct ObjectWrite {
    storage: TableStorage,
    path: String,
    buffer: Vec<u8>,
}

#[async_trait]
impl FileWrite for ObjectWrite {
    async fn write(&mut self, bytes: Bytes) -> iceberg::Result<()> {
        self.buffer.extend_from_slice(&bytes);
        Ok(())
    }

    async fn close(&mut self) -> iceberg::Result<()> {
        let bytes = Bytes::from(std::mem::take(&mut self.buffer));
        self.storage.write(&self.path, bytes).await
    }
}

fn not_found(path: &str) -> IcebergError {
    IcebergError::new(
        ErrorKind::DataInvalid,
        format!("`{path}` is not in the table's store"),
    )
}

fn into_iceberg_error(error: crate::Error) -> IcebergError {
    IcebergError::new(ErrorKind::Unexpected, "the table's store").with_source(error)
}

/// The crate's storage traits are object-safe through typetag, which asks
/// every implementation to serialize. A storage bound to this process's
/// stores serializes to its name alone and is never deserialized: a file IO
/// is rebuilt by loading the table, not from bytes.
impl Serialize for TableStorageFactory {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit_struct("TableStorageFactory")
    }
}

impl<'de> Deserialize<'de> for TableStorageFactory {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(
            "a table storage factory is bound to a process's stores and is not deserialized",
        ))
    }
}

impl Serialize for TableStorage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit_struct("TableStorage")
    }
}

impl<'de> Deserialize<'de> for TableStorage {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(
            "a table storage is bound to a process's store and is not deserialized",
        ))
    }
}

#[cfg(test)]
mod tests {
    use object_storage::{LocalStore, ObjectStore, StoreError};

    use super::*;

    /// Opens the local filesystem for a `file:///` root.
    #[derive(Debug)]
    struct LocalFactory;

    impl ExternalStoreFactory for LocalFactory {
        fn open(&self, root_uri: &str) -> object_storage::Result<Arc<dyn ObjectStore>> {
            if root_uri != "file:///" {
                return Err(StoreError::Config(format!(
                    "not a local root: `{root_uri}`"
                )));
            }
            Ok(Arc::new(LocalStore::new("/")?))
        }
    }

    fn storage_over(directory: &std::path::Path) -> TableStorage {
        let metadata_location = format!("file://{}/metadata/v1.metadata.json", directory.display());
        let store = TableStore::open("db.t", &metadata_location, &LocalFactory, None).unwrap();
        TableStorage { store }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_object_written_through_a_writer_is_read_back_by_its_location() {
        let directory = tempfile::tempdir().unwrap();
        let storage = storage_over(directory.path());
        let location = format!("file://{}/metadata/m0.avro", directory.path().display());

        let mut writer = storage.writer(&location).await.unwrap();
        writer.write(Bytes::from_static(b"mani")).await.unwrap();
        writer.write(Bytes::from_static(b"fest")).await.unwrap();
        writer.close().await.unwrap();

        assert!(storage.exists(&location).await.unwrap());
        assert_eq!(storage.metadata(&location).await.unwrap().size, 8);
        assert_eq!(storage.read(&location).await.unwrap().as_ref(), b"manifest");
        let reader = storage.reader(&location).await.unwrap();
        assert_eq!(reader.read(4..8).await.unwrap().as_ref(), b"fest");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_object_that_is_not_there_is_reported_as_missing() {
        let directory = tempfile::tempdir().unwrap();
        let storage = storage_over(directory.path());
        let location = format!("file://{}/metadata/none.avro", directory.path().display());

        assert!(!storage.exists(&location).await.unwrap());
        let error = storage.read(&location).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_location_outside_the_tables_root_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let storage = storage_over(directory.path());

        let error = storage
            .write("s3://other/metadata/m0.avro", Bytes::from_static(b"x"))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("outside"), "{error}");
    }
}
