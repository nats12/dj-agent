"""Tests for title/artist cleanup."""

from dj_agent.cleanup import (
    cleanup_artist,
    cleanup_title,
    extract_featured_artists,
    normalize_for_split,
    smart_title_case,
    split_artist_from_title,
)


class TestCleanupTitle:
    def test_html_entities(self):
        title, changes = cleanup_title("Beethoven&#39;s Aria")
        assert title == "Beethoven's Aria"
        assert any("HTML" in c for c in changes)

    def test_watermark_removal(self):
        title, _ = cleanup_title("Sandstorm (2024 Extended Mix) djsoundtop.com")
        assert "djsoundtop" not in title

    def test_file_extension_removal(self):
        title, _ = cleanup_title("Track Name.mp3")
        assert title == "Track Name"

    def test_beatport_id_removal(self):
        title, _ = cleanup_title("Track Name-62381251")
        assert title == "Track Name"

    def test_year_preserved_when_meaningful(self):
        # "Levels 2013" — could be a legit title variant, but currently
        # stripped if it looks like a bare trailing year.
        # The fix: don't strip if preceded by a common preposition.
        title, _ = cleanup_title("Class of 2013")
        assert "2013" in title  # "of 2013" should be preserved

    def test_store_id_removed(self):
        title, changes = cleanup_title("Some Track AB12345")
        assert title == "Some Track"

    def test_broad_code_not_removed(self):
        # The old pattern [A-Z]{0,3}\d{3,5} would eat "Channel 4000"
        title, _ = cleanup_title("Channel 4000")
        assert title == "Channel 4000"

    def test_master_suffix_removed(self):
        title, _ = cleanup_title("My Track MASTER")
        assert title == "My Track"

    def test_master_in_word_preserved(self):
        title, _ = cleanup_title("Grandmaster Flash")
        assert title == "Grandmaster Flash"

    def test_pn_suffix_removed(self):
        # _pn marks Platinum Notes-mastered copies (duplicates of the
        # non-_pn original); the suffix is a filename marker, not part
        # of the title.
        title, changes = cleanup_title("Somebody to Love_pn")
        assert title == "Somebody to Love"
        assert any("_pn" in c for c in changes)

    def test_pn_not_stripped_from_word_mid_title(self):
        # Only trailing "_pn" should be removed. Words that happen to
        # contain "pn" mid-string must be preserved.
        title, _ = cleanup_title("Sharpness Expanded")
        assert title == "Sharpness Expanded"

    def test_pn_stripped_before_underscore_conversion(self):
        # Titles like "Foo - Bar_MST_V4_pn" previously broke: the
        # 3+-underscore rule converted all underscores to spaces first,
        # leaving a trailing bare "pn" token that the _pn rule couldn't
        # match. Now _pn strips early and the rest is left intact.
        title, changes = cleanup_title(
            "Hugel - I adore you (Braynod & Tekzee edit)_MST_V4_pn"
        )
        assert not title.endswith("pn")           # no stray trailing pn
        assert not title.endswith("_pn")          # suffix fully removed
        assert any("_pn" in c for c in changes)


class TestArtistSplit:
    def test_basic_split(self):
        artist, title = split_artist_from_title("Charlotte de Witte - Overdrive")
        assert artist == "Charlotte de Witte"
        assert title == "Overdrive"

    def test_hyphenated_artist_not_split(self):
        artist, title = split_artist_from_title("Jay-Z - Empire State Of Mind")
        assert artist == "Jay-Z"
        assert title == "Empire State Of Mind"

    def test_no_space_hyphen_not_split(self):
        # "Some-Thing" with no spaces should NOT split
        artist, title = split_artist_from_title("Some-Thing Goes Here")
        assert artist is None
        assert "Some-Thing" in title

    def test_existing_artist_skips(self):
        artist, title = split_artist_from_title(
            "Charlotte de Witte - Overdrive", existing_artist="Charlotte de Witte"
        )
        assert artist is None

    def test_remix_in_artist_rejected(self):
        artist, title = split_artist_from_title("Original Mix - Extended")
        assert artist is None  # "Original Mix" contains "mix"


class TestFeaturedArtists:
    def test_feat_dot(self):
        main, guests = extract_featured_artists("Drake feat. Rihanna")
        assert main == "Drake"
        assert guests == ["Rihanna"]

    def test_ft(self):
        main, guests = extract_featured_artists("Artist ft Vocalist")
        assert main == "Artist"
        assert guests == ["Vocalist"]

    def test_no_feat(self):
        main, guests = extract_featured_artists("Solo Artist")
        assert main == "Solo Artist"
        assert guests == []

    def test_vs(self):
        main, guests = extract_featured_artists("Artist A vs Artist B")
        assert main == "Artist A"
        assert guests == ["Artist B"]

    def test_b2b(self):
        main, guests = extract_featured_artists("DJ Alpha b2b DJ Beta")
        assert main == "DJ Alpha"
        assert guests == ["DJ Beta"]

    def test_ampersand(self):
        main, guests = extract_featured_artists("Bicep & Hammer")
        assert main == "Bicep"
        assert guests == ["Hammer"]


class TestCleanupArtist:
    def test_html_entities(self):
        artist, changes = cleanup_artist("Beethoven&#39;s Band")
        assert artist == "Beethoven's Band"
        assert any("HTML" in c for c in changes)

    def test_trims_whitespace(self):
        artist, _ = cleanup_artist("  Artist Name  ")
        assert artist == "Artist Name"

    def test_does_not_strip_year(self):
        # A title-side rule would eat "1998"; the artist cleanup must not.
        artist, _ = cleanup_artist("Binary Finary 1998")
        assert artist == "Binary Finary 1998"


class TestPipelineOrder:
    """Regression tests for split→cleanup order in the analysis pipeline."""

    def test_numeric_title_not_stripped(self):
        # "Binary Finary - 1998" must produce artist='Binary Finary',
        # title='1998'. Previously cleanup_title stripped the year before
        # the split could see it.
        artist, title_part = split_artist_from_title("Binary Finary - 1998")
        assert artist == "Binary Finary"
        cleaned, _ = cleanup_title(title_part)
        assert cleaned == "1998"

    def test_watermark_after_split(self):
        # Artist on the left must be preserved; watermark on the right
        # must be cleaned from the title only.
        artist, title_part = split_artist_from_title(
            "Safar (FR) - Love Parade - www.electronicfre"
        )
        assert artist == "Safar (FR)"
        cleaned, changes = cleanup_title(title_part)
        assert cleaned == "Love Parade"
        assert any("watermark" in c for c in changes)

    def test_underscore_filename_splits_after_normalization(self):
        # Files like "Einmusik_Aaron_Suiss_-_Sabai" must still split
        # correctly after the pipeline reorder. The split regex needs
        # whitespace around the dash, so underscores must be converted
        # BEFORE the split — but without running title-specific rules.
        normalized = normalize_for_split("Einmusik_Aaron_Suiss_-_Sabai")
        artist, title_part = split_artist_from_title(normalized)
        assert artist == "Einmusik Aaron Suiss"
        assert title_part == "Sabai"

    def test_normalize_preserves_short_numeric_title(self):
        # Underscore normalization must not damage the numeric-title case.
        normalized = normalize_for_split("Binary_Finary_-_1998")
        artist, title_part = split_artist_from_title(normalized)
        assert artist == "Binary Finary"
        cleaned, _ = cleanup_title(title_part)
        assert cleaned == "1998"

    def test_normalize_leaves_simple_dashes_alone(self):
        # No underscores -> no change, split still works on " - ".
        assert normalize_for_split("Artist - Title") == "Artist - Title"

    def test_trailing_year_in_real_title_still_stripped(self):
        # Legitimate title-side year should still be removed by rule 10.
        artist, title_part = split_artist_from_title("Some Artist - Song Name 2013")
        assert artist == "Some Artist"
        cleaned, changes = cleanup_title(title_part)
        assert cleaned == "Song Name"
        assert any("2013" in c for c in changes)


class TestSmartTitleCase:
    def test_basic(self):
        assert smart_title_case("hello world") == "Hello World"

    def test_lowercase_connectors(self):
        result = smart_title_case("the art of mixing")
        assert result == "The Art of Mixing"

    def test_uppercase_acronyms(self):
        result = smart_title_case("dj set in nyc")
        assert result == "DJ Set in NYC"
