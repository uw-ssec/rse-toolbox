import io
import itertools
import json
import os
import zipfile
from hashlib import sha256
from typing import List

import fsspec
import pandas as pd
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers import PyPDFParser
from langchain_core.documents import Document


def open_arxiv_dataset(zip_file: str) -> pd.DataFrame:
    """
    Open arXiv dataset retrieved from Kaggle:
    https://www.kaggle.com/datasets/Cornell-University/arxiv

    Parameters
    ----------
    zip_file : str
        Path to the zip file containing the dataset,
        this can be a direct http link to the zip file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the dataset
    """
    cols = ["id", "title", "abstract", "categories"]

    with fsspec.open(zip_file) as f:
        with zipfile.ZipFile(f) as archive:
            data = []
            json_file = archive.filelist[0]
            with archive.open(json_file) as f:
                for line in io.TextIOWrapper(f, encoding="latin-1"):
                    doc = json.loads(line)
                    lst = [doc["id"], doc["title"], doc["abstract"], doc["categories"]]
                    data.append(lst)

            df_data = pd.DataFrame(data=data, columns=cols)
    return df_data


def load_pdfs_to_documents(zip_file: str, flatten: bool = False) -> List[Document]:
    """
    Load PDFs from a zip file into a list of Document objects.

    Parameters
    ----------
    zip_file : str
        Path to the zip file containing the PDFs,
        this can be a direct http link to the zip file.

    Returns
    -------
    List[Document]
        List of Document objects
    """
    with fsspec.open(zip_file) as f:
        with zipfile.ZipFile(f) as archive:
            documents = []
            # Get all the PDF files in the zip
            pdf_files = archive.filelist
            for pdf_file in pdf_files:
                # Go through each PDF file
                with archive.open(pdf_file) as f:
                    pdf_parser = PyPDFParser()
                    # Read the pdf file blob
                    blob = Blob.from_data(f.read())
                    # Use the parser to lazy parse the PDF
                    lazy_docs = pdf_parser.lazy_parse(blob)
                    document_chunks = []
                    for doc in lazy_docs:
                        fname = os.path.basename(pdf_file.filename)
                        # Hash the file name string
                        fname_hash = sha256(fname.encode("utf-8")).hexdigest()
                        # Store the source file name as metadata
                        doc.metadata.update({"source": fname, "source_id": fname_hash})
                        document_chunks.append(doc)
                    documents.append(document_chunks)

    if flatten:
        # Flatten to a single list
        documents = list(itertools.chain.from_iterable(documents))
    return documents
