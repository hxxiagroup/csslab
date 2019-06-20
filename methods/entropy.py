import os
import sys
import time
import pandas as pd
import numpy as np
from scipy.stats import entropy as _sci_entropy


'''
-------------------------------------------------
                    熵的计算
                    
1.普通属性表节点熵值
2.对于多层网络来说，计算网络节点基于某一属性的熵值(通过流量 或者 度计算)
                    
-------------------------------------------------

数据类型：

    原始数据应该有的属性
        class_type: 表示某个类型下的数量（流量）
        idx：区分(例如节点，边名称)

    数据的样式
        subgraph: 
        用来计算多层网络基于某一属性的节点熵值
            参数里面叫做graph_dic, 数据如：{"attr_1": df1, "attr_2":df2, ...},
            其中：df时DataFrame表，表示的时子网络的边数据，包含["Source","Target","Weight"],
            采用 init_with_subgraphes(),完成初始化。
                需要指定的其他参数
                    directed：表示有向网络，默认为True
                    weighted：表示加权网络，默认为True
                init_with_subgraphes(), 方法结束后，后产生infodata的数据（目前不会单独保存），见下面
                    有向图，会产生4个表，分别表示：节点入度（权），出度，总和，边信息表。
                    无向图：产生2个表，表示节点和边的信息（权）。

        infodata:
            dataframe,列名为，["Id","Attr_1","Attr_2","Attr_3",...]
            用于熵的计算。
            
            例如，节点信息表，边信息表

            其中，"Id"时必须的，其他表示类别名称
            采用 init_with_infodata()完成初始化，必须指定类别的列名，用list表示

        ent_data:
            dataframe，计算结果，包含["Id","Ent",...]的主要信息，


Entropy计算：
    raw_entropy: 
        原始的公式，手写对比scipy
 
    sci_entropy:
        scipy.stats有
           https://docs.scipy.org/doc/scipy/reference/
           generated/scipy.stats.entropy.html 
    modified_entropy:
        Liang 2012关于出行与land use文献中使用的，除以了ln(M)

    proposed_entropy:
        由于类型各个**总数据量**差异，p(x)可能需要再除以X的总量

日志；
    1.关于ent的计算完全采用sci的计算方式（不需要提前计算概率，sci会计算）
    这样的话，就没有必要将e_datas复制成p_datas, 
    

@author: 文
@date: 2018.12
@email: 245885195@qq.com
    
'''


class Entropy():

    DIRECTED_DTS = ["in","out","all","flow"]
    NON_DIRECTED_DTS = ["all","flow"]
    DEFAULT_DT = "default"

    def __init__(self):
        self.class_columns = None
        self.e_datas = {}
        self._index = "Id"
        self.init_type = None
        self._is_data_processed = False
        self._calculated = {"Ent":None,"RawEnt":None,"ModEnt":None}
    #  def transfer_edges

    def init_with_subgraphes(self,graph_dic,directed=True,weighted=True):

        if(directed):
            self.e_datas = dict.fromkeys(Entropy.DIRECTED_DTS)
        else:
            self.e_datas = dict.fromkeys(Entropy.NON_DIRECTED_DTS)

        if(not weighted):
            for ky in graph_dic.keys():
                graph_dic[ky]["Weight"] = 1

        time_1 = time.clock()
        self.nodeinfo_from_subgraphes(graph_dic,directed=directed)
        print("[process node ent]，time：{:.3f}".format(time.clock() -time_1))

        time_1 = time.clock()
        self.edgeinfo_from_subgraphes(graph_dic,directed=directed)
        print("[process edge ent]，time：{:.3f}".format(time.clock() -time_1))
        
        # 这里如果有枚举多好
        self.init_type = "subgraph_init"

    def init_with_infodata(self,e_data,class_columns=None):

        self.e_datas[Entropy.DEFAULT_DT] = e_data
        self.class_columns = class_columns

        self.init_type = "infodata_init"
        

    def nodeinfo_from_subgraphes(self,graph_dic,directed=True):
        '''
        转化子图数据到records

        目前只是有向图

        难题：
            多个图中的同一个node，如何重复定位？
        
        1. list of dict:
            node作为index？
                用桶的思想？只能使用与int类型的idx，不太好
                解决办法：用 node_to_index_mapping 来保存在表里的位置
                
            存在入数据和出数据中，节点不一致，可能节点只有入度
            采用统一的index，可以保证in，out，all中数据顺序一致（最后排序不行？）
        
        2.2.dict of dict：
            node最为dict的key，内层dict同上面，[idx,attr1,attr2]
            这样的话，最终顺序不一定，需要多次判断in keys

        创建方式：
            提前创建，就不存在判断isin了，但是节点数据量多的话，就慢了，哎，真难搞
                - dict 和 dict还不能提前知道key
                - 存在冗余，例如这个节点没有入的流量，会设置全部为0
            动态的话，python的list扩容机制同样拖慢时间啊

        暂定： 
            list of dict 然后内层的dict提前创建，同时采用index mapping的方式

        结果验证：
            对比了条件索引的方式：结果一致，速度快2倍
                遍历图的方式: 7.20s
                条件索引方式：18.9s
            对比了network中图的weighteddegree的方式：结果一致
        '''
        
        #  if(n_nodes is None):
            #  node_sets = set()
            #  for graph_i in graph_dic.values():
                #  node_sets = node_sets.union(graph_i["Source"].unique())
                #  node_sets = node_sets.union(graph_i["Target"].unique())
            #  n_nodes = len(node_sets) 

        if(self.class_columns is None):
            self.class_columns = list(graph_dic.keys())

        ret_cols = [self._index] + self.class_columns

        results = {ky_:[] for ky_ in self.e_datas.keys()}
        if("flow" in results.keys()):
            results.pop("flow") # 边信息单独计算，以免后面覆盖边的结果

        index_flag = 0
        node_index_mapping = dict()
        for attr_name,edges in graph_dic.items():
            ''' 遍历子图 '''
            for idx_ in edges.index:
                node_id_s = str(edges.ix[idx_,"Source"])
                node_id_t = str(edges.ix[idx_,"Target"])
                flow = edges.ix[idx_,"Weight"]

                # 处理两个节点
                for is_target,node in enumerate([node_id_s,node_id_t]):
                    if(node not in node_index_mapping.keys()):
                        
                        # 初始化该节点的信息表
                        for key_ in results.keys():
                            ret_data_dic = {ky_:0 for ky_ in ret_cols} 
                            ret_data_dic[self._index] = "None"
                            results[key_].append(ret_data_dic)

                        node_index = index_flag          
                        node_index_mapping[node] = node_index
                        index_flag += 1
                    else:
                        node_index  = node_index_mapping.get(node)
                
                    # 处理流量
                    if(directed):
                        if(is_target):
                            # 在一个图中会多次多为一个target和source
                            # 在多个图中同样
                            results["in"][node_index][self._index] = str(node)
                            results["in"][node_index][attr_name] += flow
                        else:
                            results["out"][node_index][self._index] = str(node)
                            results["out"][node_index][attr_name] += flow

                    results["all"][node_index][self._index] = str(node)
                    results["all"][node_index][attr_name] += flow 
           
        for data_type in results.keys():
            each_data = pd.DataFrame(results[data_type])
            each_data = each_data[each_data[self._index] != "None"]
            self.e_datas[data_type] = each_data

            #  save_file_name = "result_{}.csv"
            #  save_path = save_file_name.format(data_type)
            #  each_data.to_csv(save_path,index=False)

    
    def edgeinfo_from_subgraphes(self,graph_dic,directed=True):
        '''
        目前来看，边太多太慢了，可能是dict的索引导致
            时间：28w条边的全网络，需要20s

        result采用的结构：
            dict of dict，外层用边（source-target）作为键
                          内层包含列信息(Id,Attr_1,Attr_2,...)
            
        '''

        if(self.class_columns is None):
            self.class_columns = list(graph_dic.keys())

        source_name = "Source"
        target_name = "Target"

        ret_cols = [self._index] + self.class_columns
        result = {}
        for ty,data in graph_dic.items():
            for idx in data.index:
                s_ = data.ix[idx,source_name]
                t_ = data.ix[idx,target_name]
                
                st = "{}-{}".format(str(s_),str(t_))
                st_key = st

                if(directed):
                    if(st_key not in result.keys()):
                        result[st_key] = {self._index:st_key,source_name:s_,target_name:t_}
                else:
                    if(st_key not in result.keys()):
                        reverse_st = "{}-{}".format(str(t_),str(s_))
                        if(reverse_st in result.keys()):
                            #如果反方向的在
                            st_key = reverse_st
                        else:
                            #都没有，创建st作为键
                            result[st_key] = {self._index:st_key,source_name:s_,target_name:t_}
                # 正常来说一个网络中，边是唯一的，无向图也只有一条记录,所以下面只要=就行了
                result[st_key][ty] = data.ix[idx,"Weight"]
        
        df = pd.DataFrame(result)  #这里估计耗时多，内层用
        df = df.T
        self.e_datas["flow"] = df


    @staticmethod
    def __prob_cal(datas):
        datas = datas.divide(datas.sum(axis=1),axis=0)
        return datas

    @staticmethod
    def __clean_index(datas):
        '''把全部为0的行去掉'''
        return datas.any(axis=1)

    #  @staticmethod
    #  def __figure_out_attrs(datas):
        #  if(self.class_columns is None):
            #  try:
                #  self.class_columns = datas.columns
            #  except KeyError as e:
                #  print("KeyError，没有columns！",e)
            #  if(self._index in self.class_columns):
                #  self.class_columns.remove(self._index)
        #  return self.class_columns

    def process_data(self):
        #fillna
        for ky,data in self.e_datas.items():
            data = data.fillna(0)
            
            data = data[self.__clean_index(data[self.class_columns])]

            #  data[self.class_columns] = self.__prob_cal(data[self.class_columns]) 
            self.e_datas[ky] = data

        self._is_data_processed = True


    def raw_entropy(self):
        '''
        ## params
            @datas: DataFrame of [Id, attr_name1,attr_name2..]
            @class_columns: list, 类别的名字列表，表示所有类别
        '''

        if(not self._is_data_processed):
            self.process_data()

        def _ent(prob_df):
            '''这里在计算的时候，把包含0的行都增加了0.00000001，是考虑所有类别，结果差不大'''
            index_not_all = ~prob_df.all(axis=1)
            prob_df.ix[index_not_all] = prob_df.ix[index_not_all] + 0.0000001
            #下面是计算公式
            prob_df = prob_df * np.log(prob_df)
            prob_df = -1.0 * np.sum(prob_df,axis=1)
            return prob_df

        for ky,data in self.e_datas.items():
            self.e_datas[ky]["RawEnt"] = _ent(self.__prob_cal(data[self.class_columns]))

        self._calculated["RawEnt"] = True
            #  print("max and min: ",self.e_datas[ky]["RawEnt"].max(),self.e_datas[ky]["RawEnt"].min())

    def modified_entropy(self):
        '''
        E = - P(x)*ln(P(x)) / ln(X), X是类别数量
        '''
        COEF = np.log(len(self.class_columns))

        if(not self._calculated["Ent"]):
            self.entropy()

        for ky in self.e_datas.keys():
            self.e_datas[ky]["ModEnt"] = self.e_datas[ky]["Ent"] / COEF
        
        self._calculated["ModEnt"] = True


    def entropy(self):
        ''' 使用 scipy中熵的计算公式'''

        if(not self._is_data_processed):
            self.process_data()

        # 采用universal function的方式计算，注意看sci的源码，需要转置
        for ky in self.e_datas.keys():
            #  ret_ent = []
            #  for i in self.e_datas[ky].index:
                #  one_line_data = self.e_datas[ky].ix[i,self.class_columns].values.tolist()[:]
                #  ret_ent.append(_sci_entropy(one_line_data))

            self.e_datas[ky]["Ent"] = _sci_entropy(self.e_datas[ky][self.class_columns].T)

        self._calculated["Ent"] = True
            #  print("max and min: ",self.e_datas[ky]["RawEnt"].max(),self.e_datas[ky]["RawEnt"].min())

    def save_result(self,save_dir=None,fname_header=None,keep_infodata=True):

        save_dir = "" if save_dir is None else save_dir

        save_file_name = "ent_results_{}.csv"
        if(fname_header is not None):
            save_file_name = fname_header + "_" + save_file_name
        

        for ky,data in self.e_datas.items():
            e_col = [self._index]
            if(keep_infodata):
                e_col = e_col + self.class_columns

            save_col = e_col + list(set(data.columns.values).difference(e_col))

            save_path = os.path.join(save_dir,save_file_name.format(ky))
            # DEFAULT_DT文件名修改hash
            if(ky == Entropy.DEFAULT_DT):
                if(fname_header is None):
                    file_no = len(data) % len(Entropy.DIRECTED_DTS)
                    save_path = save_path.replace(ky,"{}_{}".format(ky,str(file_no)))
                else:
                    #  save_path = save_path.replace(ky,"{}_{}".format(ky,str(file_no)))
                    pass

            try:
                data = data[save_col]
                data.to_csv(save_path,index=False)
                print("[saved]:",save_path)
            except Exception as e:
                print("保存文件错误：{}".format(ky))
                print(e)


        #  save_file_name = "ent_results_{}.csv"
        #  for ky,data in self.p_datas.items():
            #  p_col = [self._index]
            #  if(keep_prob_data):
                #  p_col = p_col + self.class_columns

            #  save_col = p_col + list(set(data.columns.values).difference(e_col))
            #  try:
                #  data = data[save_col]
                #  save_path = os.path.join(save_dir,save_file_name.format(ky))
                #  data.to_csv(save_path,index=False)
            #  except Exception as e:
                #  print("保存文件错误：{}".format(ky))
                #  print(e)
        

def test_subgraph():
    import os

    '''
    从不同的子图来分析节点的熵
        例如，子图有步行，公交，自行车，汽车，地铁等五个网络
        熵指的就是某个网络节点，5中不同出行方式的混合程度
    graph_dic 是一个字典，key是子图的类别（名字），value是子图的边数据,DataFrame
    '''
    trans = ["Walk","Bus","Bike","Vehicle","Railway"]
    graph_dic = dict.fromkeys(trans)   # 总共5层子网络
    for each in graph_dic.keys():
        data_each = pd.DataFrame([[1,2],[2,3]], columns=["Source","Target"])
        graph_dic[each] = data_each

    ent = Entropy() #初始化一个实例

    ent.init_with_subgraphes(graph_dic,directed=True,weighted=True) # 数据初始化

    ent.entropy()  # 计算节点信息熵

    ent.modified_entropy()  #计算修正熵值，归一到[0,1]

    ent.save_result(r"./some_dir_you_like",keep_infodata=1)

def test_infodata():
    import os
    '''
    测试从infodata初始化
    '''
    class_names = ["Walk","Bus","Bike","Vehicle","Railway"]
    nodedata = [[0, 1,2,3,4,5],
                [1, 1,2,1,1,1],
                [2, 3,4,1,2,3]]
    nodedata = pd.DataFrame(nodedata, columns=["Id", *class_names])

    ent = Entropy()
    #通过infodata初始化
    ent.init_with_infodata(nodedata,class_columns=class_names)

    ent.entropy() # 计算一般的信息熵值
    ent.modified_entropy()  # 计算修正熵

    ent.save_result(r"./some_dir_you_like")



if __name__ == "__main__":
    import time
    
    time1 = time.clock()

    test_subgraph()

    test_infodata()


    print("runtime, {:.3f}".format(time.clock() - time1))





        



