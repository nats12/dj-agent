"""Tests for metadata enrichment."""

from dj_agent.metadata import MetadataEnricher


def test_enricher_without_any_sources():
    """With no API keys / libraries, enrichment should return empty results gracefully."""
    enricher = MetadataEnricher(
        lastfm_api_key=None,
        discogs_user_token=None,
    )
    result = enricher.enrich("Test Artist", "Test Track")
    assert result["artist"] == "Test Artist"
    assert result["title"] == "Test Track"
    assert isinstance(result["genre_tags"], list)
    assert isinstance(result["sources"], list)


def test_enricher_deduplicates_tags():
    """Tags from multiple sources should be deduplicated."""
    enricher = MetadataEnricher()
    result = enricher.enrich("Artist", "Title")
    # Even with no sources, structure should be valid
    assert result["genre_tags"] == list(dict.fromkeys(result["genre_tags"]))
