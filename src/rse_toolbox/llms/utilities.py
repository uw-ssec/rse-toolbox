import textwrap
from typing import Any, Dict, List

from langchain_core.documents import Document


def format_metadata(metadata: Dict[str, Any]) -> str:
    """Format the metadata of a document as a string."""
    return textwrap.dedent(
        f"""\
        ------
        Source file name: {metadata['source']}
        Page number: {metadata['page'] + 1}
        ------

        """
    )


def format_content(page_content: str) -> str:
    """Format the content of a page as a string."""
    return textwrap.dedent(
        f"""\

        {page_content}

        """
    )


def format_document(document: Document) -> str:
    """Format a document as a string including the header and the content."""
    return format_metadata(document.metadata) + format_content(document.page_content)


def format_documents_content(documents: List[Document]) -> str:
    """Format the content of a list of documents as a string."""
    return "".join([format_document(document) for document in documents])
