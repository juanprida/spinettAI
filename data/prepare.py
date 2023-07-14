"""Parse lyrics from json file."""

from typing import List

from lyricsgenius import Genius
import json
import re

# Get your token from https://docs.genius.com/
TOKEN = # Your token here
ARTISTS = [
    "Almendra",
    "Pescado Rabioso",
    "Invisible",
    "Spinetta Jade",
    "Spinetta y los Socios del Desierto",
    "Luis Alberto Spinetta"]


class LyricsDownloader:
    """Download lyrics from Genius API and save them in a json file."""

    def __init__(self, token: str, path: str = "lyrics.json"):
        """Initialize the LyricsDownloader class."""
        self.token = token
        self.path = path
        self.genius = Genius(token)
        self.artists = {}
        self.lyrics = {}

    def get_lyrics(self, artists_name: List[str]):
        """Download lyrics from Genius API and save them in a json file."""
        for artist_name in artists_name:
            if artist_name not in self.artists:
                artist = self.genius.search_artist(artist_name, sort="title")
                for song in artist.songs:
                    self.lyrics[song.title] = song.lyrics

    def save_lyrics(self):
        """Save lyrics in a json file."""
        with open(f"{self.path}", "w") as f:
            json.dump(self.lyrics, f)

class LyricsParser:
    """
    Parse lyrics from json file.
    Removes songs that are not suitable for the analysis and cleans the lyrics.
    """

    def __init__(self, path: str = "lyrics.json"):
        self.path = path
        self.songs = json.load(open(self.path))
        # sort by song title
        self.songs = dict(sorted(self.songs.items(), key=lambda x: x[0]))

    def standardize_titles(self):
        """Remove capitalization and punctuation from song titles."""
        self.songs = {
            song.lower().replace(".", "").replace("?", ""): lyrics
            for song, lyrics in self.songs.items()
        }

    def filter_songs(self, min_length=10):
        """
        Remove songs that are not suitable for the analysis.

        Parameters
        ----------
        min_length : int, optional
            Minimum length of the lyrics, by default 10.
        """
        avoid = [
            "vivo",
            "instrumental",
            "remix",
            "karaoke",
            "cover",
            "bandas eternas",
        ]
        unique_songs = {}
        for song, lyrics in self.songs.items():
            if (
                song not in unique_songs
                and not (any(word in song.lower() for word in avoid))
                and len(lyrics) >= min_length
            ):
                unique_songs[song] = lyrics

        self.songs = unique_songs

    def clean_lyrics(self):
        """Remove useless info from lyrics"""
        self.songs = {
            song: self._process_lyrics(lyrics)
            for song, lyrics in self.songs.items()
        }

    def _process_lyrics(self, lyrics: str):
        """
        Process lyrics removing unneeded information.

        Parameters
        ----------
        lyrics : str
            Lyrics to be processed.

        Returns
        -------
        str
            Processed lyrics.
        """
        # Find first \n\n and keep the text after it.
        lyrics = lyrics.split("Lyrics", 1)[1]
        # Remove the text inside [].
        lyrics = re.sub(r"\[.*?\]", "", lyrics)
        # Force the text to start with a new line.
        lyrics = lyrics.lstrip("\n")

        # Remove garbage text at the end of the lyrics.
        lyrics = lyrics.split("You might also like", 1)[0]

        # If we have multiple new lines, keep only one.
        lyrics = re.sub(r"\n+", "\n", lyrics)

        return lyrics

    def save_lyrics(self):
        """Save lyrics in a json file."""
        with open(f"{self.path}", "w") as f:
            json.dump(self.songs, f)


if __name__ == "__main__":
    print("Downloading lyrics...")
    ld = LyricsDownloader(TOKEN)
    ld.get_lyrics(ARTISTS)
    ld.save_lyrics()

    print("Parsing lyrics...")
    lp = LyricsParser()
    lp.standardize_titles()
    lp.filter_songs()
    lp.clean_lyrics()
    lp.save_lyrics()
    print("Done!")
