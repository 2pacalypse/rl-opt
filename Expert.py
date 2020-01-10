from utils import connect
class Expert:

    #for tree
    join_types = ['Nested Loop', 'Hash Join', 'Merge Join']
    scan_types = ['Seq Scan', 'Index Scan', 'Bitmap Heap Scan', 'Index Only Scan']

    #for ast
    binary_ops = {'eq', 'neq', 'lt', 'gt', 'lte', 'gte', 'like', 'nlike', 'in', 'nin'}
    unary_ops = {'exists', 'missing'}
    ternary_ops = {'between', 'not between'}

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

