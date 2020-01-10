from utils import connect

class Executor:
    def __init__(self):
        self.conn = connect()


    """
    Given an SQL query, this returns the PostgreSQL execution time in milliseconds
    """
    def get_latency_at_target(self, query, hint='', timeout=0):
        query = hint + "EXPLAIN (ANALYZE, FORMAT JSON) " + query + ";"
        cursor = self.conn.cursor()
        cursor.execute("SET statement_timeout = %s", (timeout,))
        cursor.execute(query)
        row = cursor.fetchone()
        cursor.close()
        return row[0][0]['Execution Time']    

    """
    Return the table names in the database as a list. (Used for state encoding)
    """
    def get_relations(self):
        cursor = self.conn.cursor()
        q = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name != 'queries'"
        cursor.execute(q)
        relations = []
        for table in cursor.fetchall():
            relations.append(table[0])
        cursor.close()
        return relations


    """
    Given the table names, return a dictionary of attributes of those. dict[table_name] = [col1Name, col2Name, ...]
    """
    def get_attributes(self, relations):
        attributes = dict()
        cursor = self.conn.cursor()
        for r in relations:
            q  = """select column_name from information_schema.columns where table_name = %s """
            cursor.execute(q, (r,))
            row = cursor.fetchall()
            row = [x[0] for x in row]
            attributes[r] = row
        cursor.close()
        return attributes
