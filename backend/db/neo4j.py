from neo4j import AsyncDriver, AsyncGraphDatabase

from backend.core.config import get_settings

_driver: AsyncDriver | None = None


class Neo4jConfigurationError(RuntimeError):
    pass


async def get_neo4j_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        settings = get_settings()
        if not settings.neo4j_uri or not settings.neo4j_password:
            raise Neo4jConfigurationError(
                "Neo4j credentials are missing. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
            )
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
    return _driver


async def close_neo4j_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
