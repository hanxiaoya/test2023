# 一、项目介绍
基于垂直网站数据的医药知识图谱构建  
# 二、项目运行方式
1. 配置要求：要求配置neo4j数据库及相应的python依赖包。neo4j数据库用户名密码记住，并修改相应配置文件config.py。  
2. 知识图谱数据导入：python build_medicalgraph.py，导入的数据较多，估计需要几个小时。 
# 三、说明
1. 知识图谱实体类型

| 实体类型 | 中文含义 | 实体数量 |举例 |
| :--- | :---: | :---: | :--- |
| Check | 诊断检查项目 | 3,353| 支气管造影;关节镜检查|
| Department | 医疗科目 | 54 |  整形美容科;烧伤科|
| Disease | 疾病 | 8,807 |  血栓闭塞性脉管炎;胸降主动脉动脉瘤|
| Drug | 药品 | 3,828 |  京万红痔疮膏;布林佐胺滴眼液|
| Food | 食物 | 4,870 |  番茄冲菜牛肉丸汤;竹笋炖羊肉|
| Producer | 在售药品 | 17,201 |  通药制药青霉素V钾片;青阳醋酸地塞米松片|
| Symptom | 疾病症状 | 5,998 |  乳腺组织肥厚;脑实质深部出血|
| Total | 总计 | 44,111 | 约4.4万实体量级|
   
2.知识图谱实体关系类型

| 实体关系类型 | 中文含义 | 关系数量 | 举例|
| :--- | :---: | :---: | :--- |
| belongs_to | 属于 | 8,844| <妇科,属于,妇产科>|
| common_drug | 疾病常用药品 | 14,649 | <阳强,常用,甲磺酸酚妥拉明分散片>|
| do_eat |疾病宜吃食物 | 22,238| <胸椎骨折,宜吃,黑鱼>|
| drugs_of |  药品在售药品 | 17,315| <青霉素V钾片,在售,通药制药青霉素V钾片>|
| need_check | 疾病所需检查 | 39,422| <单侧肺气肿,所需检查,支气管造影>|
| no_eat | 疾病忌吃食物 | 22,247| <唇病,忌吃,杏仁>|
| recommand_drug | 疾病推荐药品 | 59,467 | <混合痔,推荐用药,京万红痔疮膏>|
| recommand_eat | 疾病推荐食谱 | 40,221 | <鞘膜积液,推荐食谱,番茄冲菜牛肉丸汤>|
| has_symptom | 疾病症状 | 5,998 |  <早期乳腺癌,疾病症状,乳腺组织肥厚>|
| acompany_with | 疾病并发疾病 | 12,029 | <下肢交通静脉瓣膜关闭不全,并发疾病,血栓闭塞性脉管炎>|
| Total | 总计 | 294,149 | 约30万关系量级|

3. 知识图谱属性类型

| 属性类型 | 中文含义 | 举例 |
| :--- | :---: | :---: |
| name | 疾病名称 | 喘息样支气管炎 |
| desc | 疾病简介 | 又称哮喘性支气管炎... |
| cause | 疾病病因 | 常见的有合胞病毒等...|
| prevent | 预防措施 | 注意家族与患儿自身过敏史... |
| cure_lasttime | 治疗周期 | 6-12个月 |
| cure_way | 治疗方式 | "药物治疗","支持性治疗" |
| cured_prob | 治愈概率 | 95% |
| easy_get | 疾病易感人群 | 无特定的人群 |


# 四、Cypher 图数据库查询语句

查看所有节点

```
MATCH (n) RETURN n
```

可视化知识图谱元数据图（Ontology）
```
CALL db.schema.visualization()
```


百日咳 和 血常规 的关系
```
MATCH (p:Disease {name:'百日咳'})-[]->(q:Check {name:'血常规'}) RETURN p,q
```

所有疾病和对应的检查
```
MATCH (p:Disease)-[r:need_check]->(q:Check) RETURN p,q
```

删除所有节点
```
MATCH (n) DETACH DELETE n
```


# 五、图数据挖掘

Graph Apps --> Graph Apps Gallery --> Graph Data Science Playground-->安装

节点重要性：Page Rank

社群发现：Louvain

图嵌入：Node2Vec

