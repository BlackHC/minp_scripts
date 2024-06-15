import blackhc.project.script
import blackhc.project
import sqlite3
import pickle


class EmbeddingCache:
    def __init__(self, db_path=f'{blackhc.project.project_dir}/embeddings_cache.db'):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """Create the embeddings table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                input_text TEXT PRIMARY KEY,
                embedding BLOB
            )
            ''')
            conn.commit()

    def has(self, text):
        """Check if the embedding for the given text is already in the database."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('SELECT 1 FROM embeddings WHERE input_text = ?', (text,))
            return c.fetchone() is not None

    def get(self, text):
        """Retrieve the embedding for the given text from the database."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('SELECT embedding FROM embeddings WHERE input_text = ?', (text,))
            result = c.fetchone()
            return pickle.loads(result[0]) if result else None
        
    def get_batch(self, texts):
        for text in texts:
            embedding = self.get(text)
            yield embedding

    def put(self, text, embedding):
        """Store the embedding for the given text in the database."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('INSERT OR REPLACE INTO embeddings (input_text, embedding) VALUES (?, ?)', 
                      (text, pickle.dumps(embedding)))
            conn.commit()

    def put_batch(self, texts, embeddings):
        """Store a batch of embeddings for the given texts in the database."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Prepare data for batch insertion
            data_to_insert = [(text, pickle.dumps(embedding)) for text, embedding in zip(texts, embeddings)]
            # Execute batch insert
            c.executemany('INSERT OR REPLACE INTO embeddings (input_text, embedding) VALUES (?, ?)', data_to_insert)
            conn.commit()
