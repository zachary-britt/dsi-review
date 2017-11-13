import numpy as np

def mat_mult():
    A = [[3,4,1],[3,1,1]]
    B = [[1,0,0],[0,1,0],[0,0,1]]
    if np.array(A).shape[1] != np.array(B).shape[0]:
        return 'No can do'
    # A x B = ?
    return [[sum([z[0]*z[1] for z in zip(A_row,B_col)]) for B_col in zip(*B)] for A_row  in A]

def sql_query():
    query = '''
    SELECT AVG(sale_price/beds), AVG(sale_price/sqft)
    FROM houses
    '''
    query2 = '''
    SELECT neighborhood
    FROM houses
    WHERE type = 'single_family'
    GROUP BY neighborhood
    ORDER BY count(*) DESC
    LIMIT 1
    '''

def q3():
    pass
    #1: poisson
    #2: normal
    #3: poisson


if __name__ == '__main__':
    print(mat_mult())
