import sqlite3
from DatabaseManager import DatabaseManager
from pipelines.BeatmapIngestionPipeline import BeatmapIngestionPipeline

if __name__ == "__main__":
    # clear map_portfolios and map_stats table and then insert
    """
    1. connect to db
    2. delete all from map_portfolios and map_stats
    3. reparse all maps
    4. insert new data into map_portfolios and map_stats
    """
    conn = sqlite3.connect("data/osu_profiler.db")
    cursor = conn.cursor()
    # drop tables maps, map_portfolios, map_stats
    cursor.execute("DROP TABLE IF EXISTS maps")
    cursor.execute("DROP TABLE IF EXISTS map_portfolios")
    cursor.execute("DROP TABLE IF EXISTS map_stats")
    conn.commit()
    conn.close()
    db = DatabaseManager("data/osu_profiler.db")
    db.init_db()
    BeatmapIngestionPipeline(db, "data/maps").bulk_ingest()
