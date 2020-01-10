import torch

PGHOST = "localhost"
PGDATABASE = "imdb"
#PGUSER = "murat"
#PGPASSWORD = "123456"


#for tree
join_types = ['Nested Loop', 'Hash Join',]
# 'Merge Join']
scan_types = ['Seq Scan', 'Index Scan', 'Bitmap Heap Scan', 'Index Only Scan']

#for ast
binary_ops = {'eq', 'neq', 'lt', 'gt', 'lte', 'gte', 'like', 'nlike', 'in', 'nin'}
unary_ops = {'exists', 'missing'}
ternary_ops = {'between', 'not between'}

#schema
relations =  ['aka_title', 'title', 'company_type', 'aka_name', 'role_type', 'cast_info', 'comp_cast_type', 'complete_cast', 'info_type', 'name', 'char_name', 'company_name', 'kind_type', 'link_type', 'keyword', 'movie_keyword', 'movie_link', 'movie_info', 'movie_companies', 'person_info', 'movie_info_idx']
columns =  {'aka_title': ['episode_of_id', 'season_nr', 'episode_nr', 'kind_id', 'id', 'production_year', 'movie_id', 'md5sum', 'title', 'imdb_index', 'phonetic_code', 'note'], 'title': ['episode_of_id', 'season_nr', 'episode_nr', 'production_year', 'id', 'imdb_id', 'kind_id', 'md5sum', 'title', 'imdb_index', 'phonetic_code', 'series_years'], 'company_type': ['id', 'kind'], 'aka_name': ['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], 'role_type': ['id', 'role'], 'cast_info': ['person_role_id', 'person_id', 'movie_id', 'id', 'role_id', 'nr_order', 'note'], 'comp_cast_type': ['id', 'kind'], 'complete_cast': ['id', 'movie_id', 'subject_id', 'status_id'], 'info_type': ['id', 'info'], 'name': ['imdb_id', 'id', 'imdb_index', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum', 'name'], 'char_name': ['imdb_id', 'id', 'imdb_index', 'md5sum', 'name_pcode_nf', 'surname_pcode', 'name'], 'company_name': ['imdb_id', 'id', 'country_code', 'md5sum', 'name_pcode_nf', 'name_pcode_sf', 'name'], 'kind_type': ['id', 'kind'], 'link_type': ['id', 'link'], 'keyword': ['id', 'keyword', 'phonetic_code'], 'movie_keyword': ['id', 'movie_id', 'keyword_id'], 'movie_link': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'], 'movie_info': ['id', 'movie_id', 'info_type_id', 'info', 'note'], 'movie_companies': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'], 'person_info': ['id', 'person_id', 'info_type_id', 'info', 'note'], 'movie_info_idx': ['id', 'movie_id', 'info_type_id', 'info', 'note']}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
