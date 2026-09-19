from pathlib import Path


def load_text_file(file_path):
    """
    Load a text file and return its contents.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_text_files(directory):
    """
    Load all .txt files from a directory.
    """

    documents = []

    for file_path in Path(directory).glob("*.txt"):
        text = load_text_file(file_path)

        documents.append({
            "source": str(file_path),
            "text": text
        })

    return documents