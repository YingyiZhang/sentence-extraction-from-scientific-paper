'''
find FE related to methods and problems from database
'''

'''
 problem
1、Showing limitation or lack of past work(intro)
2、Showing the main problem in the field(intro)
3、Showing the aim of the paper(intro)
4、Showing the limitation of the research(intro)

method
1、Showing brief introduction to the methodology(intro)
2、Description of the process(method)
3、Using methods used in past work(method)
4、Showing methodology used in past work(method)

'''
problem = ['Showing limitation or lack of past work','Showing the main problem in the field','Showing the aim of the paper','Showing the limitation of the research']
method = ['Showing brief introduction to the methodology','Description of the process','Using methods used in past work','Showing methodology used in past work']

FE_methods = []
FE_problems = []
with open(r"FES\watsuki-2021-EACL\NERdepparseLMI\FE-database\CL.introduction","r") as fr:
    for line in fr.readlines():
        sp = line.strip().split("\t")
        cate = sp[0]
        fe = sp[1]
        if cate in problem:
            FE_problems.append(fe)
        if cate in method:
            FE_methods.append(fe)

with open(r"FES\Iwatsuki-2021-EACL\NERdepparseLMI\FE-database\CL.methods","r") as fr:
    for line in fr.readlines():
        sp = line.strip().split("\t")
        cate = sp[0]
        fe = sp[1]
        if cate in problem:
            FE_problems.append(fe)
        if cate in method:
            FE_methods.append(fe)

with open("FES/FES_problem","w") as fw:
    for value in FE_problems:
        fw.write(value+"\n")

with open("FES/FES_method","w") as fw:
    for value in FE_methods:
        fw.write(value+"\n")

exit()