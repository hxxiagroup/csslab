import pymysql

'''
目的：
    利用pymysql用来连接Server数据库，获取数据表数据等操作

方法：
    * 连接数据库 - 创建Connection实例
    * 获取数据表数据 - get_table_data
    * 获取数据表字段 - get_column

备注：
    需要数据库相关的功能补充
    对于pymysql的功能需要再补充
'''


class Connection():
    '''
        通过连接本机数据库，可以直接查询数据库中的表格数据等。

    '''

    def __init__(self, server_name, user_name, psw, db_name):
        '''

        :param server_name: the server of database
        :param user_name: user's name for logining in your database
        :param psw: password for logining in your database
        :param db_name: database name
        '''
        self.server = server_name
        self.user = user_name
        self.password = psw
        self.dbname = db_name
        self.con = pymysql.connect(self.server, self.user, self.password, self.dbname)
        cur = self.con.cursor()
        cur.execute('select name from sysobjects where xtype=\'u\'')
        alltable = cur.fetchall()
        self.tables = [alltable[i][0] for i in range(len(alltable))]
        cur.close()

    def get_colunm(self, q_table):
        '''
        #获取表格的字段
        :param q_table: 查询表的名称
        :return: colunmn
        '''
        cur = self.con.cursor()
        cur.execute('select name from syscolumns where id=object_id(\'%s\')' % (q_table))
        allcol = cur.fetchall()
        tablecol = [allcol[i][0] for i in range(len(allcol))]
        cur.close()
        return tablecol

    def get_table_data(self, q_table, q_range=None):
        '''
        获取数据表数据
        :param q_table: 查询表格名称
        :param q_range: 查询数据范围，前 * 行，默认为全部
        :return: 数据
        '''
        if q_range is None:
            rg = '*'
        else:
            rg = 'Top ' + str(q_range) + ' * '
        cur = self.con.cursor()
        cur.execute('select %s from %s' % (rg, q_table))
        res = cur.fetchall()
        cur.close()
        return res

    def close(self):
        self.con.close()
