# coding: utf-8
from py2neo import Graph, Node

class Config(object):
    def __init__(self):
        self.data_path = './data/medical2.json'
        self.graph = Graph('http://localhost:7474', auth=('neo4j', '12345678'), name='neo4j')

        # 抽取到的实体字典
        self.drugs_path = './data/dict/drugs.json'
        self.foods_path = './data/dict/foods.json'
        self.checks_path = './data/dict/checks.json'
        self.departments_path = './data/dict/departments.json'
        self.producers_path = './data/dict/producers.json'
        self.diseases_path = './data/dict/diseases.json'
        self.symptoms_path = './data/dict/symptoms.json'

        # 抽取到的三元组
        self.rels_department_path = './data/triples/rels_department.json'
        self.rels_noteat_path = './data/triples/rels_noteat.json'
        self.rels_doeat_path = './data/triples/rels_doeat.json'
        self.rels_recommandeat_path = './data/triples/rels_recommandeat.json'
        self.rels_commonddrug_path = './data/triples/rels_commonddrug.json'
        self.rels_recommanddrug_path = './data/triples/rels_recommanddrug.json'
        self.rels_check_path = './data/triples/rels_check.json'
        self.rels_drug_producer_path = './data/triples/rels_drug_producer.json'
        self.rels_symptom_path = './data/triples/rels_symptom.json'
        self.rels_acompany_path = './data/triples/rels_acompany.json'
        self.rels_category_path = './data/triples/rels_category.json'
