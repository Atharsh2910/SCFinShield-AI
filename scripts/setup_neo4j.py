import asyncio

from backend.db.neo4j import close_neo4j_driver, get_neo4j_driver


NEO4J_SCHEMA_QUERIES = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_risk_score IF NOT EXISTS FOR (e:Entity) ON (e.risk_score)",
    "CREATE CONSTRAINT invoice_id IF NOT EXISTS FOR (i:Invoice) REQUIRE i.id IS UNIQUE",
    "CREATE INDEX invoice_number IF NOT EXISTS FOR (i:Invoice) ON (i.invoice_number)",
]


async def setup_neo4j() -> None:
    driver = await get_neo4j_driver()
    async with driver.session() as session:
        for query in NEO4J_SCHEMA_QUERIES:
            await session.run(query)
    await close_neo4j_driver()


if __name__ == "__main__":
    asyncio.run(setup_neo4j())
