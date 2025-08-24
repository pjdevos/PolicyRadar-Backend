from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import feedparser
from SPARQLWrapper import SPARQLWrapper, JSON
import logging
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DocItem:
    """A standardized data structure for a policy document."""
    id: str
    source: str
    doc_type: str
    title: str
    summary: str
    body_text: str
    url: str
    published: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    language: Optional[str] = 'en'
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self):
        """Serializes the DocItem to a JSON string."""
        # A simple version, assuming `published` is a string or None.
        # For datetime objects, you'd need a custom encoder.
        data = self.__dict__.copy()
        return json.dumps(data, ensure_ascii=False)

class IngestionService:
    """A service to ingest documents from various sources."""

    def __init__(self, settings):
        self.settings = settings

    def ingest_euractiv(self) -> List[DocItem]:
        """Fetches and parses documents from the EURACTIV RSS feed."""
        logging.info(f"Ingesting from EURACTIV RSS feed: {self.settings.ingestion.EURACTIV_RSS_URL}")

        try:
            # Add a common browser user-agent to avoid 403 Forbidden errors
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            feed = feedparser.parse(self.settings.ingestion.EURACTIV_RSS_URL, agent=user_agent)

            if feed.bozo:
                # Bozo flag is set if the feed is not well-formed
                logging.error(f"Failed to parse EURACTIV feed. Error: {feed.bozo_exception}")
                return []

            docs = []
            for entry in feed.entries:
                try:
                    # Convert published time tuple to ISO 8601 string
                    published_time = time.strftime('%Y-%m-%dT%H:%M:%SZ', entry.published_parsed) if hasattr(entry, 'published_parsed') else None

                    # Use summary as body_text as full text is not in RSS
                    body_text = entry.summary

                    doc = DocItem(
                        id=entry.id,
                        source='EURACTIV',
                        doc_type='News',
                        title=entry.title,
                        url=entry.link,
                        summary=entry.summary,
                        body_text=body_text,
                        published=published_time,
                        topics=[tag.term for tag in entry.tags] if hasattr(entry, 'tags') else [],
                        language=entry.get('language', 'en')
                    )
                    docs.append(doc)
                except Exception as e:
                    logging.error(f"Error processing EURACTIV entry '{entry.get('link', 'N/A')}': {e}")

            logging.info(f"Successfully ingested {len(docs)} documents from EURACTIV.")
            return docs

        except Exception as e:
            logging.error(f"An unexpected error occurred during EURACTIV ingestion: {e}")
            return []

    def ingest_eurlex(self) -> List[DocItem]:
        """Fetches and parses documents from the EUR-Lex SPARQL endpoint."""
        logging.info(f"Ingesting from EUR-Lex SPARQL endpoint: {self.settings.ingestion.EUR_LEX_SPARQL_ENDPOINT}")
        endpoint_url = self.settings.ingestion.EUR_LEX_SPARQL_ENDPOINT

        # Calculate the date for 365 days ago to use in the filter
        cutoff_date = datetime.now() - timedelta(days=365)
        cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')

        # This query retrieves recent regulations and directives from EUR-Lex.
        query = f"""
            PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            SELECT DISTINCT ?work ?title ?date ?url
            WHERE {{
                ?work a cdm:work .

                ?work cdm:work_title ?title_node .
                FILTER(lang(?title_node) = 'en')
                BIND(STR(?title_node) AS ?title)

                ?work cdm:work_date_document ?date .

                # Filter for documents published after our calculated cutoff date
                FILTER(?date > "{cutoff_date_str}"^^xsd:date)

                # Get the HTML URL for the English version
                ?work cdm:work_is_realized_by_expression ?expression .
                ?expression cdm:expression_language <http://publications.europa.eu/resource/authority/language/ENG> .
                ?expression cdm:expression_manifested_by_manifestation ?manifestation .
                ?manifestation cdm:manifestation_type "html" .
                ?manifestation cdm:manifestation_url ?url .
            }}
            ORDER BY DESC(?date)
            LIMIT 50
        """

        try:
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

            docs = []
            for result in results["results"]["bindings"]:
                doc_id_uri = result.get("work", {}).get("value")
                if not doc_id_uri:
                    continue

                # Create a unique ID from the document's URI
                doc_id = doc_id_uri.split('/')[-1]

                # The body_text is not available from SPARQL, so we leave it empty.
                # A more advanced implementation could scrape the URL.
                summary = f"EU legal document published on {result.get('date', {}).get('value', 'N/A')}."

                doc = DocItem(
                    id=f"eurlex_{doc_id}",
                    source='EUR-Lex',
                    doc_type='Legal',
                    title=result.get("title", {}).get("value", "No title provided"),
                    url=result.get("url", {}).get("value"),
                    summary=summary,
                    body_text="", # Full text requires scraping the URL
                    published=result.get("date", {}).get("value"),
                    topics=['EU Legislation', 'Regulation', 'Directive'], # Generic topics
                    language='en'
                )
                docs.append(doc)

            logging.info(f"Successfully ingested {len(docs)} documents from EUR-Lex.")
            return docs

        except Exception as e:
            logging.error(f"An error occurred during EUR-Lex ingestion: {e}")
            return []

    def run_ingestion(self):
        """Runs the full ingestion pipeline."""
        logging.info("Starting data ingestion run...")

        all_docs = []
        logging.warning("EURACTIV ingestion is temporarily disabled due to access being blocked (403 Forbidden).")
        # all_docs.extend(self.ingest_euractiv())
        all_docs.extend(self.ingest_eurlex())

        logging.info(f"Ingested a total of {len(all_docs)} documents.")

        # Save documents to JSONL file
        data_dir = self.settings.database.DATA_DIR
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / "items.jsonl"

        # Overwrite the file with new data
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in all_docs:
                f.write(doc.to_json() + '\n')

        logging.info(f"Saved ingested documents to {output_path}")

        return str(output_path)
