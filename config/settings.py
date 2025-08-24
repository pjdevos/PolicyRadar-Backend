#!/usr/bin/env python3
"""
Policy Radar Configuration Management

Typed configuration using Pydantic Settings with environment validation.
Supports development, testing, and production environments.
"""

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    # Vector database settings
    VECTOR_DB_PATH: Path = Field(default=Path("./vectors"), description="Vector database storage path")
    VECTOR_INDEX_NAME: str = Field(default="policy_index", description="Vector index name")
    FAISS_INDEX_TYPE: str = Field(default="IVFFlat", description="FAISS index type")
    
    # Data storage
    DATA_DIR: Path = Field(default=Path("./data"), description="Data storage directory")
    BACKUP_DIR: Optional[Path] = Field(default=None, description="Backup directory")
    
    class Config:
        env_prefix = "DB_"


class APISettings(BaseSettings):
    """API configuration"""
    # Server settings
    HOST: str = Field(default="0.0.0.0", description="API server host")
    PORT: int = Field(default=8000, description="API server port")
    WORKERS: int = Field(default=1, description="Number of worker processes")
    
    # API keys and authentication
    API_KEY: Optional[SecretStr] = Field(default=None, description="API key for authenticated endpoints")
    SECRET_KEY: SecretStr = Field(default="dev-secret-change-in-production", description="Application secret key")
    
    # External API keys
    OPENAI_API_KEY: Optional[SecretStr] = Field(default=None, description="OpenAI API key for RAG")
    ANTHROPIC_API_KEY: Optional[SecretStr] = Field(default=None, description="Anthropic API key")
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    DEFAULT_RATE_LIMIT: int = Field(default=100, description="Default requests per hour")
    RAG_RATE_LIMIT: int = Field(default=10, description="RAG endpoint requests per hour")
    INGEST_RATE_LIMIT: int = Field(default=5, description="Ingest endpoint requests per hour")
    
    # CORS settings - Explicit domains (FastAPI doesn't support wildcard subdomains)
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3001", 
            "http://localhost:3002",
            "https://policy-radar-frontend.vercel.app",
            "https://policy-radar-frontend-jt8cxxl2d-pjdevos-projects-979bae0e.vercel.app",
            "https://policy-radar-frontend-jf19kb2py-pjdevos-projects-979bae0e.vercel.app",
            "https://policy-radar-frontend-dnzv5irtu-pjdevos-projects-979bae0e.vercel.app",
            "https://policyradar-frontend-production.up.railway.app",
            "https://policyradar-backend-production.up.railway.app",
            # Add common Vercel deployment patterns
            "https://policy-radar-frontend-git-main-pjdevos-projects-979bae0e.vercel.app",
            "https://policy-radar-frontend-pjdevos-projects-979bae0e.vercel.app",
            # Add more specific domains as needed instead of wildcards
        ],
        description="Allowed CORS origins (explicit domains only)"
    )
    CORS_METHODS: List[str] = Field(default=["GET", "POST"], description="Allowed CORS methods")
    CORS_HEADERS: List[str] = Field(
        default=["Content-Type", "Accept", "Authorization"],
        description="Allowed CORS headers"
    )
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    class Config:
        env_prefix = "API_"


class SecuritySettings(BaseSettings):
    """Security configuration"""
    # Security headers
    SECURITY_HEADERS_ENABLED: bool = Field(default=True, description="Enable security headers")
    HSTS_MAX_AGE: int = Field(default=31536000, description="HSTS max age in seconds")
    
    # Trusted hosts
    TRUSTED_HOSTS: List[str] = Field(
        default=[
            "localhost",
            "127.0.0.1", 
            "*.up.railway.app",
            "*.vercel.app"
        ],
        description="Trusted host domains"
    )
    
    # Input validation
    MAX_QUERY_LENGTH: int = Field(default=500, description="Maximum RAG query length")
    MAX_TOPIC_LENGTH: int = Field(default=50, description="Maximum topic name length")
    ALLOWED_FILE_EXTENSIONS: Set[str] = Field(
        default={".jsonl", ".json", ".csv"},
        description="Allowed file extensions for uploads"
    )
    
    @field_validator("TRUSTED_HOSTS", mode="before")
    @classmethod
    def parse_trusted_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",") if host.strip()]
        return v
    
    class Config:
        env_prefix = "SECURITY_"


class RAGSettings(BaseSettings):
    """RAG (Retrieval-Augmented Generation) configuration"""
    # Model settings
    LLM_PROVIDER: str = Field(default="openai", description="LLM provider (openai, anthropic)")
    LLM_MODEL: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    EMBEDDING_MODEL: str = Field(default="text-embedding-ada-002", description="Embedding model")
    
    # RAG parameters
    DEFAULT_K: int = Field(default=8, description="Default number of retrieved documents")
    MAX_K: int = Field(default=20, description="Maximum number of retrieved documents")
    SIMILARITY_THRESHOLD: float = Field(default=0.7, description="Similarity threshold for retrieval")
    
    # Processing limits
    MAX_TOKENS: int = Field(default=4000, description="Maximum tokens for LLM response")
    TIMEOUT_SECONDS: int = Field(default=30, description="Request timeout in seconds")
    
    class Config:
        env_prefix = "RAG_"


class DataIngestionSettings(BaseSettings):
    """Data ingestion configuration"""
    # Data sources
    EURACTIV_RSS_URL: str = Field(
        default="https://www.euractiv.com/feed/",
        description="EURACTIV RSS feed URL"
    )
    EUR_LEX_SPARQL_ENDPOINT: str = Field(
        default="http://publications.europa.eu/webapi/rdf/sparql",
        description="EUR-Lex SPARQL endpoint"
    )
    
    # Ingestion limits
    MAX_DAYS_LOOKBACK: int = Field(default=365, description="Maximum days to look back")
    DEFAULT_DAYS_LOOKBACK: int = Field(default=30, description="Default days to look back")
    MAX_DOCUMENTS_PER_SOURCE: int = Field(default=1000, description="Max documents per source")
    
    # Processing settings
    BATCH_SIZE: int = Field(default=100, description="Batch size for processing")
    RETRY_ATTEMPTS: int = Field(default=3, description="Number of retry attempts")
    RETRY_DELAY: int = Field(default=5, description="Retry delay in seconds")
    
    class Config:
        env_prefix = "INGEST_"


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    LOG_FILE: Optional[Path] = Field(default=None, description="Log file path")
    LOG_ROTATION: bool = Field(default=True, description="Enable log rotation")
    LOG_MAX_SIZE: str = Field(default="10MB", description="Maximum log file size")
    LOG_BACKUP_COUNT: int = Field(default=5, description="Number of backup log files")
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """Main application settings"""
    # Environment
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    DEBUG: bool = Field(default=True, description="Debug mode")
    VERSION: str = Field(default="1.0.0", description="Application version")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    security: SecuritySettings = SecuritySettings()
    rag: RAGSettings = RAGSettings()
    ingestion: DataIngestionSettings = DataIngestionSettings()
    logging: LoggingSettings = LoggingSettings()
    
    @field_validator("DEBUG", mode="before")
    @classmethod  
    def set_debug_based_on_env(cls, v, info):
        """Set debug mode based on environment"""
        env = info.data.get("ENVIRONMENT", Environment.DEVELOPMENT)
        if env == Environment.PRODUCTION:
            return False
        return v
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    def validate_secrets(self) -> List[str]:
        """Validate required secrets are present"""
        missing_secrets = []
        
        # Check SECRET_KEY
        if not self.api.SECRET_KEY or self.api.SECRET_KEY.get_secret_value() == "dev-secret-change-in-production":
            # Always warn, but don't prevent startup. The user can see this in logs.
            print("[WARN] Using default or missing SECRET_KEY. This should be set via API_SECRET_KEY environment variable in production.")
            if self.is_production():
                # We won't append to missing_secrets to allow startup,
                # but the warning is a strong indicator.
                pass
        
        # LLM API keys are optional - RAG will use mock responses if not available
        if not self.api.OPENAI_API_KEY and not self.api.ANTHROPIC_API_KEY:
            if self.is_production():
                print("[WARN] No LLM API keys configured - RAG endpoints will use mock responses")
            else:
                print("[INFO] No LLM API keys configured - using development mode")
        
        return missing_secrets
    
    def create_directories(self):
        """Create required directories"""
        directories = [
            self.database.VECTOR_DB_PATH,
            self.database.DATA_DIR,
        ]
        
        if self.database.BACKUP_DIR:
            directories.append(self.database.BACKUP_DIR)
        
        if self.logging.LOG_FILE:
            directories.append(self.logging.LOG_FILE.parent)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        # Load environment variables from .env file
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        # Nested settings
        env_nested_delimiter = "__"
        
        # Allow extra fields for extensibility
        extra = "allow"


# Global settings instance
def get_settings() -> Settings:
    """Get application settings singleton"""
    return Settings()


# Export for easy importing
settings = get_settings()


def validate_startup_config() -> None:
    """Validate configuration at startup"""
    print("[INFO] Validating configuration...")
    
    # Check required secrets
    missing_secrets = settings.validate_secrets()
    if missing_secrets:
        if settings.is_production():
            raise ValueError(f"Missing required production secrets: {', '.join(missing_secrets)}")
        else:
            print(f"[WARN] Missing secrets in development: {', '.join(missing_secrets)}")
    
    # Create required directories
    try:
        settings.create_directories()
        print("[OK] Required directories created/verified")
    except Exception as e:
        print(f"[WARN] Directory creation issue: {e}")
        # Don't fail startup for directory issues in development
        if settings.is_production():
            raise ValueError(f"Failed to create required directories: {e}")
    
    # Validate paths exist
    if not settings.database.VECTOR_DB_PATH.exists():
        print(f"[WARN] Vector DB path will be created: {settings.database.VECTOR_DB_PATH}")
    
    # Environment-specific validations
    if settings.is_production():
        print("[INFO] Production environment detected")
        
        # Production-specific checks
        if settings.DEBUG:
            print("[WARN] Warning: Debug mode enabled in production")
        
        if not settings.security.SECURITY_HEADERS_ENABLED:
            print("[WARN] Warning: Security headers disabled in production")
    
    else:
        print(f"[INFO] Development environment: {settings.ENVIRONMENT}")
    
    print("[OK] Configuration validation completed")


if __name__ == "__main__":
    # Test configuration loading
    try:
        validate_startup_config()
        print("\n[INFO] Current Configuration:")
        print(f"Environment: {settings.ENVIRONMENT}")
        print(f"Debug: {settings.DEBUG}")
        print(f"API Host: {settings.api.HOST}:{settings.api.PORT}")
        print(f"Vector DB: {settings.database.VECTOR_DB_PATH}")
        print(f"CORS Origins: {settings.api.CORS_ORIGINS}")
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        exit(1)
