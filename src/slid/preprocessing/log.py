"""Log pre-processing."""

import regex as re
from more_itertools import unique_everseen


NUMBERS_REGEX = re.compile(
    r"(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))"
)
URL_REGEX = r"​(https?:\/\/)?(www\.)[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)|(https?:\/\/)?(www\.)?(?!ww)[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)"
SECTION_CLEANUP_REGEX = r"(^\s?section\s*end.*cleanup\sfile\svariable)(.+)((?:\n.+)+)(section\s*start.*cleanup\sfile\svariable.*)"
SECTION_UPLOAD_ON_FAILURE = r"(^\s?section\s*end.*upload\sartifacts)(.+)((?:\n.+)+)(section\s*start.*upload\sartifacts.*)"


def clean(log: str):
    """Clean build job log data."""

    log = "".join(log.splitlines(keepends=True)[::-1])  # reverse text lines
    log = fix_encoding(log)
    log = re.sub(URL_REGEX, " <URL>", log)
    log = re.sub(r"(\/.*?\.[\w:]+)", " <FILEPATH>", log)
    log = re.sub(r"(\/(\w|-)*)+", " <DIRPATH>", log)
    log = re.sub(r"(\d+(?:\.\d+)+)s", " <DURATION>", log)
    log = re.sub(r"(v?\d+(?:\.\d+)+)", " <VERSION>", log)
    log = re.sub(r"[^a-zA-Z0-9<>\r\n]+\t*", " ", log)  # remove unknown characters
    log = re.sub(r"[0-9][0-9]+", "", log)  # remove numbers with two digits and more
    log = re.sub(r"[0-9a-fx]{5,40}", "<ID>", log)  # replace commits and IDs
    log = re.sub(r"0K|(\d[ \t]*)+m", "", log)  # remove ansi characters
    log = re.sub("\t", " ", log)  # remove blank tabs
    log = re.sub(r" [a-zA-Z] ", " ", log)  # remove trailing single character
    log = re.sub(r" +", " ", log)  # remove blank spaces
    log = re.sub(r" > ", " ", log)
    log = re.sub(r" < ", " ", log)
    log = re.sub(r"\n\s*\n", "\n", log, flags=re.M)  # remove blank lines
    log = re.sub(SECTION_CLEANUP_REGEX, "", log, flags=re.M)
    log = re.sub(SECTION_UPLOAD_ON_FAILURE, "", log, flags=re.M)
    log = re.sub(
        r"^\s?section\s(start|end).*$", "", log, flags=re.M
    )  # remove section titles
    log = re.sub(
        r"^ ", "", log, flags=re.M
    )  # remove trailing whitespace at sentence start
    log = re.sub(
        r"([A-Za-z]+[\d@]+[\w@]*|[\d@]+[A-Za-z]+[\w@]*)", "<ID>", log
    )  # remove remaining IDs
    log = re.sub(
        r"\s[a-zA-Z]$", "\n", log, flags=re.M
    )  # remove remaining single characters
    log = re.sub(
        r"(?<!(exit|status) code:? )\d", "<NUMBER>", log
    )  # remove single numbers that are not a status code

    # remove duplicate lines and pass statements of test logs
    lines = log.splitlines(keepends=True)
    lines = filter(lambda l: not l.startswith("pass "), lines)
    log = "".join(unique_everseen(lines))

    return log


def fix_encoding(text):
    try:
        decoded_text = text.encode("utf-8").decode("utf-8")
    except UnicodeDecodeError:
        decoded_text = text
    return decoded_text


def replace_numbers(text, replace_with="<NUMBER>"):
    """Replace all numbers in ``text`` str with ``replace_with`` str."""
    return NUMBERS_REGEX.sub(replace_with, text)
