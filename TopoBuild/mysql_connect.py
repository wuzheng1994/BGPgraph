# encoding: utf-8
import pymysql


class Data_from_mysql():

# 实现数据从mysql中取出,并处在字典中,对于多余的ASN再查mysql数据库。

    def __init__(self, hostname, username, database):
    
        conn = pymysql.connect(
        host=hostname,  # 与本地数据库建立连接，可以尝试其他电脑的ip地址
        port=3306,  # 端口可写可不写，默认3306
        user=username,
        password="123456",
        database=database,  # 选择一个库，关键字可以简化为db
        charset="utf8mb4"
        )
        self.cursor = conn.cursor()  # cursor=pymysql.cursors.DictCursor
        self.asn_info_dict = {}
        self.conn = conn

    def fetdata_from_mysql(self, asnset):
        sql = "SELECT asn, latitude, longitude, asrank FROM `caida_asn1` WHERE asn in ({})".format(",".join(asnset))
        self.cursor.execute(sql)
        res = self.cursor.fetchall()
        return res

    def finsh_fetch(self):
     if self.cursor:
        self.cursor.close()
        self.conn.close()

    def fetch_asnset(self, asndiff):

        res = self.fetdata_from_mysql(asndiff)
        data = dict((r[0], tuple(r[1:])) for r in res)
        unknown = asndiff - data.keys() # unknown lookups
        unknown_dict = {asn:(0, 0, "0")  for asn in unknown}
        self.asn_info_dict.update(data)
        if unknown:
            self.asn_info_dict.update(unknown_dict)
            # print('self.asn_info_dict:', self.asn_info_dict)
            # print("info:", asn_info_dict)
            # self.finsh_fetch()
    @property
    def get_asn_info(self):
        return self.asn_info_dict

if __name__ == "__main__":
    database = Data_from_mysql("192.168.1.79", username="root", database = "Stream")
    asn_set = ["7497",'13335','1','2', '-1']
    asn_info_dict = {}

    res = database.fetch_asnset(asn_set)
    print("res:",res)
    # print("info:", info)
    
