# coding: utf-8
import os
import json
import threading
from py2neo import Graph, Node
from tqdm import tqdm
from config import Config

class MedicalGraph:
    def __init__(self, config):
        super(MedicalGraph, self).__init__()
        self.config = config
        self.g = config.graph

        # 共7类节点
        self.drugs = [] # 药品
        self.foods = [] #　食物
        self.checks = [] # 检查
        self.departments = [] #科室
        self.producers = [] #药企厂商
        self.diseases = [] #疾病
        self.symptoms = []#症状

        self.disease_infos = []#疾病信息

        # 节点实体关系
        self.rels_department = [] #　科室－科室关系
        self.rels_noteat = [] # 疾病－忌吃食物关系
        self.rels_doeat = [] # 疾病－宜吃食物关系
        self.rels_recommandeat = [] # 疾病－推荐吃食物关系
        self.rels_commonddrug = [] # 疾病－通用药品关系
        self.rels_recommanddrug = [] # 疾病－热门药品关系
        self.rels_check = [] # 疾病－检查关系
        self.rels_drug_producer = [] # 厂商－药物关系

        self.rels_symptom = [] #疾病-症状关系
        self.rels_acompany = [] # 疾病-并发病关系
        self.rels_category = [] #　疾病-科室之间的关系

    '''读取文件，抽取实体、三元组'''
    def extract_triples(self):
        count = 0
        for data in open(self.config.data_path, 'r', encoding='utf-8'):
            disease_dict = {}
            count += 1
            print(count)
            data_json = json.loads(data)
            disease = data_json['name']
            disease_dict['name'] = disease
            self.diseases.append(disease)
            disease_dict['desc'] = ''
            disease_dict['prevent'] = ''
            disease_dict['cause'] = ''
            disease_dict['easy_get'] = ''
            disease_dict['cure_department'] = ''
            disease_dict['cure_way'] = ''
            disease_dict['cure_lasttime'] = ''
            disease_dict['symptom'] = ''
            disease_dict['cured_prob'] = ''

            if 'symptom' in data_json:
                self.symptoms += data_json['symptom']
                for symptom in data_json['symptom']:
                    self.rels_symptom.append([disease, 'has_symptom', symptom, 'Disease', 'Symptom', '症状'])

            if 'acompany' in data_json:
                for acompany in data_json['acompany']:
                    self.rels_acompany.append([disease, 'acompany_with', acompany, 'Disease', 'Disease', '并发症'])
                    self.diseases.append(acompany)

            if 'desc' in data_json:
                disease_dict['desc'] = data_json['desc']

            if 'prevent' in data_json:
                disease_dict['prevent'] = data_json['prevent']

            if 'cause' in data_json:
                disease_dict['cause'] = data_json['cause']

            if 'get_prob' in data_json:
                disease_dict['get_prob'] = data_json['get_prob']

            if 'easy_get' in data_json:
                disease_dict['easy_get'] = data_json['easy_get']

            if 'cure_department' in data_json:
                cure_department = data_json['cure_department']
                if len(cure_department) == 1:
                     self.rels_category.append([disease, 'cure_department', cure_department[0], 'Disease', 'Department', '所属科室'])
                if len(cure_department) == 2:
                    big = cure_department[0]
                    small = cure_department[1]
                    self.rels_department.append([small, 'belongs_to', big, 'Department', 'Department', '属于'])
                    self.rels_category.append([disease, 'cure_department', small, 'Disease', 'Department', '所属科室'])

                disease_dict['cure_department'] = cure_department
                self.departments += cure_department

            if 'cure_way' in data_json:
                disease_dict['cure_way'] = data_json['cure_way']

            if  'cure_lasttime' in data_json:
                disease_dict['cure_lasttime'] = data_json['cure_lasttime']

            if 'cured_prob' in data_json:
                disease_dict['cured_prob'] = data_json['cured_prob']

            if 'common_drug' in data_json:
                common_drug = data_json['common_drug']
                for drug in common_drug:
                    self.rels_commonddrug.append([disease, 'has_common_drug', drug, 'Disease', 'Drug', '常用药品'])
                self.drugs += common_drug

            if 'recommand_drug' in data_json:
                recommand_drug = data_json['recommand_drug']
                self.drugs += recommand_drug
                for drug in recommand_drug:
                    self.rels_recommanddrug.append([disease, 'recommand_drug', drug, 'Disease', 'Drug', '好评药品'])

            if 'not_eat' in data_json:
                not_eat = data_json['not_eat']
                for _not in not_eat:
                    self.rels_noteat.append([disease, 'not_eat', _not, 'Disease', 'Food', '忌吃'])
                self.foods += not_eat

            if 'do_eat' in data_json:
                do_eat = data_json['do_eat']
                for _do in do_eat:
                    self.rels_doeat.append([disease, 'do_eat', _do, 'Disease', 'Food', '宜吃'])
                self.foods += do_eat

            if 'recommand_eat' in data_json:
                recommand_eat = data_json['recommand_eat']
                for _recommand in recommand_eat:
                    self.rels_recommandeat.append([disease, 'recommand_eat', _recommand, 'Disease', 'Food', '推荐食谱'])
                self.foods += recommand_eat

            if 'check' in data_json:
                check = data_json['check']
                for _check in check:
                    self.rels_check.append([disease, 'need_check', _check, 'Disease', 'Check', '诊断检查'])
                self.checks += check

            if 'drug_detail' in data_json:
                drug_detail = data_json['drug_detail']
                producer = [i.split('(')[0] for i in drug_detail]
                self.rels_drug_producer += [[i.split('(')[0], 'production', i.split('(')[-1].replace(')', ''), 'Producer', 'Drug', '生产药品'] for i in drug_detail]
                self.producers += producer

            self.disease_infos.append(disease_dict)
        return set(self.drugs), set(self.foods), set(self.checks), set(self.departments), set(self.producers), set(self.symptoms), set(self.diseases), self.disease_infos,\
               self.deduplicate(self.rels_check), self.deduplicate(self.rels_recommandeat), self.deduplicate(self.rels_noteat), self.deduplicate(self.rels_doeat), \
               self.deduplicate(self.rels_department), self.deduplicate(self.rels_commonddrug), self.deduplicate(self.rels_drug_producer), self.deduplicate(self.rels_recommanddrug),\
               self.deduplicate(self.rels_symptom), self.deduplicate(self.rels_acompany), self.deduplicate(self.rels_category)

    '''关系去重函数'''
    def deduplicate(self, rels_old):
        rels_new = []
        for each in rels_old:
            if each not in rels_new:
                rels_new.append(each)
        return rels_new

    '''建立节点'''
    def create_node(self, nodes, label):
        count = 0
        try:
            for node_name in nodes:
                node = Node(label, name=node_name)
                self.g.create(node)
                count += 1
                print('创建第{}个{}实体 {}'.format(count, label, node_name))
        except:
            pass
        print('共创建 {} 个{}实体'.format(len(nodes), label))
        return

    def set_attributes(self, node_infos, label):
        # for node in tqdm(node_infos):
        for node in node_infos:
            name = node['name']
            # del node['name']
            for k, v in node.items():
                if k in ['cure_department', 'cure_way']:
                    sql = """MATCH (n:{label})
                            WHERE n.name='{name}'
                            set n.{k}={v}""".format(label=label, name=name.replace("'", ""), k=k, v=v)
                else:
                    sql = """MATCH (n:{label}) 
                            WHERE n.name='{name}'
                            set n.{k}='{v}'""".format(label=label, name=name.replace("'", ""), k=k,
                                                              v=v.replace("'", "").replace("\n", ""))
                try:
                    self.g.run(sql)
                except Exception as e:
                    print(e)
                    print(sql)


    # '''创建知识图谱中心疾病的节点'''
    # def create_diseases_nodes(self, disease_infos):
    #     count = 0
    #     for disease_dict in disease_infos:
    #         try:
    #             node = Node("Disease",
    #                         name=disease_dict['name'],
    #                         desc=disease_dict['desc'],
    #                         prevent=disease_dict['prevent'],
    #                         cause=disease_dict['cause'],
    #                         easy_get=disease_dict['easy_get'],
    #                         cure_lasttime=disease_dict['cure_lasttime'],
    #                         cure_department=disease_dict['cure_department'],
    #                         cure_way=disease_dict['cure_way'],
    #                         cured_prob=disease_dict['cured_prob'])
    #             self.g.create(node)
    #             count += 1
    #             print('创建疾病实体：', disease_dict['name'])
    #         except:
    #             pass
    #     print('共创建 {} 个疾病实体'.format(count))
    #     return

    '''创建实体关联边'''
    def create_relationship(self, edges):
        count = 0
        # 去重处理
        edges = self.deduplicate(edges)
        all = len(edges)
        for edge in edges:
            p, q = edge[0], edge[2]
            p_type, q_type = edge[3], edge[4]
            rel_type, rel_name = edge[1], edge[5]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                p_type, q_type, p, q, rel_type, rel_name)
            print(query)
            try:
                self.g.run(query)
                count += 1
                print("创建关系{}-{}->{}, 第{}, 共{}个" % (p, rel_type, q, count, all))
            except Exception as e:
                print(e)
        return

    '''创建知识图谱实体节点类型schema'''
    def create_graphnodes(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug, rels_drug_producer, rels_recommanddrug,rels_symptom, rels_acompany, rels_category = self.extract_triples()
        # self.create_diseases_nodes(disease_infos)
        self.create_node(Diseases, 'Disease')
        self.create_node(Drugs, 'Drug')
        self.create_node(Foods, 'Food')
        self.create_node(Checks, 'Check')
        self.create_node(Departments, 'Department')
        self.create_node(Producers, 'Producer')
        self.create_node(Symptoms, 'Symptom')
        return

    '''创建知识图谱实体节点属性schema'''
    def set_node_attributes(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug, rels_drug_producer, rels_recommanddrug, rels_symptom, rels_acompany, rels_category = self.extract_triples()
        # self.set_attributes(disease_infos, "Disease")
        t = threading.Thread(target=self.set_attributes,args=(disease_infos, "Disease"))
        t.setDaemon(False)
        t.start()

    '''创建实体关系边'''
    def create_graphrels(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug, rels_drug_producer, rels_recommanddrug,rels_symptom, rels_acompany, rels_category = self.extract_triples()
        self.create_relationship(rels_recommandeat)
        self.create_relationship(rels_noteat)
        self.create_relationship(rels_doeat)
        self.create_relationship(rels_department)
        self.create_relationship(rels_commonddrug)
        self.create_relationship(rels_drug_producer)
        self.create_relationship(rels_recommanddrug)
        self.create_relationship(rels_check)
        self.create_relationship(rels_symptom)
        self.create_relationship(rels_acompany)
        self.create_relationship(rels_category)



    '''导出数据'''
    def export_data(self, data, export_path):
        with open(export_path, 'w+', encoding='utf-8') as f:
            f.write('\n'.join(list(data)))
        f.close()
    def export_triples_data(self, data, export_path):
        with open(export_path, 'w+', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        f.close()

    def export_entity_relations(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug, rels_drug_producer, rels_recommanddrug, rels_symptom, rels_acompany, rels_category = self.extract_triples()
        self.export_data(Drugs, self.config.drugs_path)
        self.export_data(Foods, self.config.foods_path)
        self.export_data(Checks, self.config.checks_path)
        self.export_data(Departments, self.config.departments_path)
        self.export_data(Producers, self.config.producers_path)
        self.export_data(Diseases, self.config.diseases_path)
        self.export_data(Symptoms, self.config.symptoms_path)

        self.export_triples_data(rels_department, self.config.rels_department_path)
        self.export_triples_data(rels_noteat, self.config.rels_noteat_path)
        self.export_triples_data(rels_doeat, self.config.rels_doeat_path)
        self.export_triples_data(rels_recommandeat, self.config.rels_recommandeat_path)
        self.export_triples_data(rels_commonddrug, self.config.rels_commonddrug_path)
        self.export_triples_data(rels_recommanddrug, self.config.rels_recommanddrug_path)
        self.export_triples_data(rels_check, self.config.rels_check_path)
        self.export_triples_data(rels_drug_producer, self.config.rels_drug_producer_path)
        self.export_triples_data(rels_symptom, self.config.rels_symptom_path)
        self.export_triples_data(rels_acompany, self.config.rels_acompany_path)
        self.export_triples_data(rels_category, self.config.rels_category_path)
        return


if __name__ == '__main__':
    # import sys
    # print(sys.getdefaultencoding())
    config = Config()
    handler = MedicalGraph(config)

    # 删除所有实体和关系
    cypher = 'MATCH (n) DETACH DELETE n'
    handler.g.run(cypher)

    print("step1:导入图谱节点中")
    handler.create_graphnodes() # 创建实体节点
    print("step2:导入图谱边中")
    handler.create_graphrels() # 创建实体关系
    print("step3:导入图谱属性中")
    handler.set_node_attributes() # 创建实体属性


    # handler.export_entity_relations()  # 导出数据
