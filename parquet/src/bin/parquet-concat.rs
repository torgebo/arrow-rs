// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Binary that concatenates the column data of one or more parquet files
//!
//! # Install
//!
//! `parquet-concat` can be installed using `cargo`:
//! ```
//! cargo install parquet --features=cli
//! ```
//! After this `parquet-concat` should be available:
//! ```
//! parquet-concat out.parquet a.parquet b.parquet
//! ```
//!
//! The binary can also be built from the source code and run as follows:
//! ```
//! cargo run --features=cli --bin parquet-concat out.parquet a.parquet b.parquet
//! ```
//!
//! Note: this does not currently support preserving the page index or bloom filters
//!

use clap::Parser;
use parquet::bloom_filter::Sbbf;
use parquet::column::writer::ColumnCloseResult;
use parquet::errors::{ParquetError, Result};
use parquet::file::metadata::{ColumnChunkMetaData, PageIndexPolicy, ParquetMetaDataReader};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::ChunkReader;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type;
use std::fs::File;
use std::sync::Arc;

#[derive(Debug, Parser)]
#[clap(author, version)]
/// Concatenates one or more parquet files
struct Args {
    /// Path to output
    output: String,

    /// Path to input files
    input: Vec<String>,
}

fn read_bloom_filter<R: ChunkReader>(column: &ColumnChunkMetaData, input: &R) -> Option<Sbbf> {
    Sbbf::read_from_column_chunk(column, input).ok().flatten()
}

fn schema_from_file(path: &String) -> Result<Arc<Type>> {
    let reader = File::open(path)?;
    let metadata = ParquetMetaDataReader::new()
        .with_page_index_policy(PageIndexPolicy::Optional)
        .parse_and_finish(&reader)?;
    Ok::<Arc<parquet::schema::types::Type>, ParquetError>(
        metadata.file_metadata().schema_descr().root_schema_ptr(),
    )
}

impl Args {
    fn run(&self) -> Result<()> {
        if self.input.is_empty() {
            return Err(ParquetError::General(
                "Must provide at least one input file".into(),
            ));
        }

        // Compare schemas in a first pass to make sure they all match
        let expected = schema_from_file(&self.input[0])?;
        let other_schemas = self.input[1..].iter().map(|x| (x, schema_from_file(x)));
        for res_other in other_schemas {
            let path = res_other.0;
            let actual = res_other.1?;
            if actual != expected {
                return Err(ParquetError::General(format!(
                    "inputs must have the same schema: file {path}: expected schema {expected:#?} got {actual:#?}"
                )));
            }
        }

        // Given equal schemas, Copy to output file
        let output = File::create(&self.output)?;
        let props = Arc::new(WriterProperties::builder().build());
        let mut writer = SerializedFileWriter::new(output, expected, props)?;

        self.input
            .iter()
            .map(|x| {
                let input = File::open(x)?;
                let metadata = ParquetMetaDataReader::new()
                    .with_page_index_policy(PageIndexPolicy::Optional)
                    .parse_and_finish(&input)?;
                let column_indexes = metadata.column_index();
                let offset_indexes = metadata.offset_index();
                for (rg_idx, rg) in metadata.row_groups().iter().enumerate() {
                    let rg_column_indexes = column_indexes.and_then(|ci| ci.get(rg_idx));
                    let rg_offset_indexes = offset_indexes.and_then(|oi| oi.get(rg_idx));

                    let mut rg_out = writer.next_row_group()?;
                    for (col_idx, column) in rg.columns().iter().enumerate() {
                        let bloom_filter = read_bloom_filter(column, &input);
                        let column_index =
                            rg_column_indexes.and_then(|row| row.get(col_idx)).cloned();
                        let offset_index =
                            rg_offset_indexes.and_then(|row| row.get(col_idx)).cloned();
                        let result = ColumnCloseResult {
                            bytes_written: column.compressed_size() as _,
                            rows_written: rg.num_rows() as _,
                            metadata: column.clone(),
                            bloom_filter,
                            column_index,
                            offset_index,
                        };
                        rg_out.append_column(&input, result)?;
                    }
                    rg_out.close()?;
                }
                Ok(())
            })
            .collect::<Result<Vec<_>>>()?;

        writer.close()?;

        Ok(())
    }
}

fn main() -> Result<()> {
    Args::parse().run()
}

#[cfg(test)]
mod tests {

    use super::*;
    use arrow::error::ArrowError;
    use arrow_array::{Int32Array, Int64Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
    use std::clone::Clone;
    use std::sync::Arc;
    use tempfile::TempDir;

    type ArrowResult<T> = Result<T, ArrowError>;

    #[test]
    #[should_panic]
    fn test_0_cli_arguments_is_invalid() {
        Args::try_parse_from(vec![""]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_single_cli_argument_is_invalid() {
        Args::try_parse_from(vec!["parquet-concat"]).unwrap();
    }

    #[test]
    fn test_need_at_least_one_input_file() -> Result<()> {
        let tmpf = TempDir::new().unwrap();
        let out_name = tmpf
            .path()
            .join("out.parquet")
            .into_os_string()
            .into_string()
            .expect("retrieved os_string as string");
        let args =
            Args::try_parse_from(vec!["parquet-concat", &out_name]).expect("parse arguments");

        let exp_err_msg = "Must provide at least one input file";

        match args.run() {
            Err(ParquetError::General(msg)) if msg == exp_err_msg => {}
            _ => panic!("expected ParquetError::General(Must provide at least one input file)"),
        }
        Ok(())
    }

    #[test]
    fn test_observes_if_input_files_do_not_exist_produces_error_of_type_external() -> Result<()> {
        let tmpf = TempDir::new().expect("create TempDir");
        let out_name = tmpf.path().join("out.parquet");

        let in1_name = tmpf.path().join("in1.parquet");
        let in2_name = tmpf.path().join("in2.parquet");
        let in3_name = tmpf.path().join("in3.parquet");

        let inputs = vec![
            in1_name.to_str().unwrap().to_string(),
            in2_name.to_str().unwrap().to_string(),
            in3_name.to_str().unwrap().to_string(),
        ];

        let args = Args {
            output: out_name.to_str().unwrap().to_string(),
            input: inputs,
        };

        let o = args.run();

        match o {
            Err(ParquetError::External(_)) => (),
            _ => panic!("expected External ParquetError"),
        }

        Ok(())
    }

    #[test]
    fn test_concatenate_3_files() -> ArrowResult<()> {
        let num_files = 3;

        let tmpf = TempDir::new().unwrap();
        let res = run_concat_n_files(&tmpf, num_files, DatasetOne::to_rcb);

        match res {
            Ok(()) => (),
            Err(e) => panic!("expected Ok(()), got {}", e),
        }

        let reader = File::open(tmpf.path().join("out.parquet"))?;
        let mut expected = Vec::<DatasetOne>::with_capacity(3 * 16);
        for _ind in 0..num_files {
            expected.extend(DatasetOne::get());
        }

        let batch_reader: ParquetRecordBatchReader =
            ParquetRecordBatchReader::try_new(reader, 1000000).expect("reads out.parquet");

        let batches: ArrowResult<Vec<RecordBatch>> =
            batch_reader.into_iter().collect::<ArrowResult<Vec<_>>>();

        batches.map(|_t| ())
    }

    #[test]
    fn test_concatenate_3000_files() -> Result<()> {
        let num_files = 3000;
        let tmpf = TempDir::new().unwrap();
        run_concat_n_files(&tmpf, num_files, DatasetOne::to_rcb)
    }

    #[test]
    fn test_concatenate_2_files_different_schemas() -> Result<()> {
        let tmpf = TempDir::new().unwrap();
        let out_name = tmpf.path().join("out.parquet");
        let inputs: Vec<String> = vec![
            tmpf.path().join("in1.parquet").display().to_string(),
            tmpf.path().join("in2.parquet").display().to_string(),
        ];

        DatasetOne::to_rcb(&inputs[0]);
        DatasetTwo::to_rcb(&inputs[1]);

        let args = Args {
            output: out_name.to_str().unwrap().to_string(),
            input: inputs,
        };

        match args.run() {
            Err(ParquetError::General(msg)) => {
                assert!(msg.starts_with("inputs must have the same schema"));
            }
            _ => panic!("expected ParquetError::General"),
        };

        Ok(())
    }

    /// First dataset has differing `length` column in i32
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct DatasetOne {
        pub length: i32,
        pub annotation: String,
    }

    impl DatasetOne {
        pub fn get_schema() -> Arc<Schema> {
            Arc::new(Schema::new(vec![
                Field::new("total_length", DataType::Int32, false),
                Field::new("animal", DataType::LargeUtf8, false),
            ]))
        }

        pub fn get_recordbatch() -> RecordBatch {
            let schema = Self::get_schema();

            let col2_data = Arc::new(LargeStringArray::from_iter_values([
                "mouse", "cat", "horse", "mouse", "cat", "horse", "mouse", "cat", "mouse", "cat",
                "horse", "mouse", "cat", "horse", "mouse", "cat",
            ])) as Arc<dyn arrow_array::Array>;
            let col1_data = Arc::new(Int32Array::from_iter_values(0..(col2_data.len() as i32)))
                as Arc<dyn arrow_array::Array>;

            assert_eq!(
                // defensive check
                col1_data.len(),
                col2_data.len(),
                "columns are of equal length"
            );

            RecordBatch::try_new(schema.clone(), vec![col1_data, col2_data]).unwrap()
        }

        fn to_rcb(path: &str) {
            let record_batch = Self::get_recordbatch();
            let mut buffer = File::create(path).expect("create file at {path}");
            let mut writer = ArrowWriter::try_new(&mut buffer, record_batch.schema(), None)
                .expect("opened file writer");
            writer.write(&record_batch).expect("wrote to file");

            writer.close().expect("closed writer");
        }

        fn from_recordbatch(record_batch: &RecordBatch) -> Vec<DatasetOne> {
            let column1 = record_batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("failed to downcast");
            let column2 = record_batch
                .column(1)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("failed to downcast");

            let mut ds: Vec<DatasetOne> = vec![];
            for (ii, ss) in column1.iter().zip(column2.iter()) {
                ds.push(DatasetOne {
                    length: ii.expect("to be number"),
                    annotation: ss.expect("to be string").to_string(),
                });
            }
            ds
        }

        fn get() -> Vec<DatasetOne> {
            let rb = Self::get_recordbatch();
            Self::from_recordbatch(&rb)
        }
    }

    /// First dataset has differing `length` column in i64
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct DatasetTwo {
        pub length: i64,
        pub annotation: String,
    }

    impl DatasetTwo {
        pub fn get_schema() -> Arc<Schema> {
            Arc::new(Schema::new(vec![
                Field::new("length", DataType::Int64, false),
                Field::new("String", DataType::LargeUtf8, false),
            ]))
        }

        pub fn get_recordbatch() -> RecordBatch {
            let schema = Self::get_schema();

            let col2_data = Arc::new(LargeStringArray::from_iter_values([
                "mouse", "cat", "horse", "mouse", "cat", "horse", "mouse", "cat", "mouse", "cat",
                "horse", "mouse", "cat", "horse", "mouse", "cat",
            ])) as Arc<dyn arrow_array::Array>;

            let col1_data = Arc::new(Int64Array::from_iter_values(0..(col2_data.len() as i64)))
                as Arc<dyn arrow_array::Array>;

            assert_eq!(
                // defensive check
                col1_data.len(),
                col2_data.len(),
                "columns are of equal length"
            );

            RecordBatch::try_new(schema.clone(), vec![col1_data, col2_data])
                .expect("create recordbatch")
        }

        fn to_rcb(path: &str) {
            let record_batch = Self::get_recordbatch();
            let mut buffer = File::create(path).expect("create file at {path}");
            let mut writer = ArrowWriter::try_new(&mut buffer, record_batch.schema(), None)
                .expect("opened file writer");
            writer.write(&record_batch).expect("wrote to file");

            writer.close().expect("closed writer");
        }
    }

    /// Create `num_files` files of type `DatasetOne`
    /// and then run `parquet-concat` on them
    fn run_concat_n_files(tmpf: &TempDir, num_files: usize, to_rcb: impl Fn(&str)) -> Result<()> {
        let out_name = tmpf.path().join("out.parquet");

        let mut inputs = Vec::<String>::with_capacity(num_files);
        for file_index in 1..=num_files {
            let in_name = tmpf.path().join(format!("in{file_index}.parquet"));
            let file_name = in_name.display().to_string();
            to_rcb(&file_name);
            inputs.push(file_name);
        }

        let args = Args {
            //output: out_name.to_str().unwrap().to_string(),
            output: out_name.display().to_string(),
            input: inputs,
        };

        args.run()
    }
}
