from utils import connect
class Expert:

    def __init__(self):
        self.conn = connect()

    """
    Given an SQL query, this returns the generated PostgreSQL plan as a dict.
    """
    def get_expert_plan(self, query, hint='', timeout=0):
        query = hint + "EXPLAIN (FORMAT JSON) " + query + ";"
        cursor = self.conn.cursor()
        cursor.execute("SET statement_timeout = %s", (timeout,))
        cursor.execute(query)
        row = cursor.fetchone()
        cursor.close()
        return row[0][0]

