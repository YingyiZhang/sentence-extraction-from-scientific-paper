from treelib import Node, Tree
import json



def constructDependencyTree(dependency, tokens, token_lemma, entity_len):
    '''
    :param dependency:[["advmod", "15-robust", "14-more"], ["amod", "16-methodology", "15-robust"], ["det", "16-methodology", "13-a"], ["nsubjpass", "18-required", "16-methodology"], ["auxpass", "18-required", "17-is"], ["mark", "18-required", "12-that"], ["nsubj", "3-show", "1-We"], ["aux", "3-show", "2-will"], ["ccomp", "3-show", "18-required"], ["punct", "3-show", "19-."], ["ccomp", "3-show", "9-insufficient"], ["amod", "7-solutions", "5-conventional"], ["compound", "7-solutions", "6-MDP"], ["conj:and", "9-insufficient", "18-required"], ["mark", "9-insufficient", "4-that"], ["nsubj", "9-insufficient", "7-solutions"], ["cop", "9-insufficient", "8-are"], ["punct", "9-insufficient", "10-,"], ["cc", "9-insufficient", "11-and"]]
    :return: constructed tree{headid:{de_id:feature, de_id:feature}, headid:{de_id:feature, de_id:feature}}
    '''
    '''turn dependency list to dict'''
    tree_dict = {}
    tree = Tree()
    tree_lemma = Tree()
    for item in dependency:
        feature = item[0]
        head_id = int(item[1].split("-")[0])# the first word start with 1
        dependency_id = int(item[2].split("-")[0])# the first word
        if head_id in tree_dict:
            childs = tree_dict[head_id]
            childs[dependency_id] = feature
            tree_dict[head_id] = childs
        else:
            tree_dict[head_id] = {dependency_id:feature}

    '''find the root word'''
    roots = []
    for head_1, childs_1 in tree_dict.items():
        judge = 0
        for head_2, childs_2 in tree_dict.items():
            if head_1 in childs_2:
                judge=1
                break
        if judge==0:
            roots.append(head_1)

    '''construct the tree'''
    #add root node
    def iterature(tree_dict, tokens, token_lemma, parents_set):
        '''
        add childs algorithm
        :param tree_dict:
        :param tokens:
        :param token_lemma:
        :param parents_set:
        :return:
        '''
        for head_id, childs in tree_dict.items():
            if head_id in parents_set:
                for child_id, feature in childs.items():
                    child_token = tokens[child_id-1]
                    child_token_lemma = token_lemma[child_id-1]
                    if not tree.contains(child_id):
                        tree.create_node(child_token, child_id, parent=head_id, data=feature)
                        tree_lemma.create_node(child_token_lemma, child_id, parent=head_id, data=feature)
                    if child_id in tree_dict:
                        parents_set.add(child_id)
        return parents_set, tree, tree_lemma

    trees = []
    trees_lemma = []
    for root in roots:
        root_token = tokens[root-1]
        root_token_lemma = token_lemma[root-1]
        tree.create_node(root_token, root)
        tree_lemma.create_node(root_token_lemma, root, data='root')
        parents_set = set()
        parents_set.add(root)
        root_quantity = len(tree_dict)
        # add childs node
        temp = Tree()
        count = 0
        if entity_len>1:
            while len(parents_set)<root_quantity and count<100:
                count = count+1
                parents_set, tree, tree_lemma = iterature(tree_dict, tokens, token_lemma, parents_set)
            else:
                parents_set, tree, tree_lemma = iterature(tree_dict, tokens, token_lemma, parents_set)
        print ()
        trees.append(tree)
        trees_lemma.append(tree_lemma)
        tree = Tree()
        tree_lemma = Tree()

    return trees, trees_lemma

def levelOrder(tree):
    """
    :type root: Node
    :rtype: List[List[int]]
    """
    root = tree.root
    root = tree.get_node(root)
    if not root:
        return []
    que = []
    res = {}
    que.append([root,root])
    layer = 0
    while len(que):
        l = len(que)
        sub = []
        for i in range(l):
            current = que.pop(0)
            sub.append(current)
            for child in tree.children(current[0].identifier):
                que.append([child, current[0]])
        res[layer] = sub
        layer = layer+1
    return res

def containsRootToken(pattern_tree, tokens):
    has_pattern_word = False
    pattern_root = pattern_tree.get_node(pattern_tree.root).tag
    tokens = set(tokens)
    if pattern_root in tokens:
        has_pattern_word = True
    return has_pattern_word

def containsToken(pattern_tree, tokens):
    pattern_nodes = pattern_tree.all_nodes()
    tokens = set(tokens)
    ''' first, if word in pattern_tree is existed in the tokens; true: continue, false: stop'''
    has_pattern_word = True

    count = 0
    for pattern_node in pattern_nodes:
        pattern_node_tag = pattern_node.tag
        pattern_node_tag = pattern_node_tag.replace("{+METHOD}","").replace("{+TASK}","")

        if pattern_node_tag in tokens:
            count = count+1
    if count<len(pattern_nodes)-1:
        has_pattern_word = False
    return has_pattern_word

def containsTokenAll(pattern_tree, tokens):
    pattern_nodes = pattern_tree.all_nodes()
    tokens = set(tokens)
    ''' first, if word in pattern_tree is existed in the tokens; true: continue, false: stop'''
    has_pattern_word = True
    count = 0
    for pattern_node in pattern_nodes:
        pattern_node_tag = pattern_node.tag
        pattern_node_tag = pattern_node_tag.replace("{+METHOD}","").replace("{+TASK}","")
        if pattern_node_tag in tokens:
            count = count+1
    if count<len(pattern_nodes):
        has_pattern_word = False
    return has_pattern_word

def getSeedTree(dependency_tree, pattern_tree):
    '''
    Find the subtree containing the pattern from the sentence tree

    Step:
    1. Get all nodes of the sentence tree; The root node of the pattern subtree and the corresponding words ;
    2. If the ID of a node of the sentence tree is the same as the ID of the root node of the pattern subtree, the subtree corresponding to the root node is returned

    :param dependency_tree:
    :param pattern_tree:
    :return:
    '''
    #should consider that if has more than one subTree

    dependency_sub_trees = []
    dependency_nodes = dependency_tree.all_nodes()
    pattern_root = pattern_tree.get_node(pattern_tree.root)
    pattern_root_identifier = pattern_root.identifier
    for dependency_node in dependency_nodes:
        dependency_node_id = dependency_node.identifier
        if dependency_node_id == pattern_root_identifier:
            if not dependency_node.is_leaf():  # need to have childs
                dependency_sub_tree = dependency_tree.subtree(dependency_node_id)  # find the sub tree
                dependency_sub_tree.update_node(dependency_sub_tree.root)
                dependency_sub_trees.append(dependency_sub_tree)
            else:#If the node of phrase is leaf
                dependency_sub_tree = dependency_tree.subtree(dependency_node_id)  # find the sub tree
                dependency_sub_tree.update_node(dependency_sub_tree.root)
                dependency_sub_trees.append(dependency_sub_tree)
    return dependency_sub_trees

def getSubTree(dependency_tree, pattern_tree):
    '''
    :param dependency_tree:句子树
    :param pattern_tree:pattern子树
    :return:
    '''
    #should consider that if has more than one subTree
    dependency_sub_trees = []
    dependency_nodes = dependency_tree.all_nodes()
    pattern_root = pattern_tree.get_node(pattern_tree.root)
    pattern_root_tag = pattern_root.tag
    for dependency_node in dependency_nodes:
        dependency_node_tag = dependency_node.tag.replace("{+METHOD}","").replace("{+TASK}","")
        dependency_node_id = dependency_node.identifier
        if dependency_node_tag == pattern_root_tag:
            if not dependency_node.is_leaf():  # need to have childs
                dependency_sub_tree = dependency_tree.subtree(dependency_node_id)  # find the sub tree
                dependency_sub_tree.update_node(dependency_sub_tree.root)
                dependency_sub_trees.append(dependency_sub_tree)
    #the feature of root change to None
    #more than one sub_tree,   tree list
    return dependency_sub_trees

def extractSentence(iter_idx, pattern_tree_layerOrder, dependency_sub_tree_layerOrder, pattern_tree, dependency_sub_tree, category):
    count = 0

    for pattern_layer, pattern_values in pattern_tree_layerOrder.items():
        dependency_values = dependency_sub_tree_layerOrder[pattern_layer]
        for pattern_value in pattern_values:
            pattern_child = pattern_value[0]
            pattern_child_tag = pattern_child.tag
            pattern_child_feature = pattern_child.data
            for dependency_value in dependency_values:
                dependency_child = dependency_value[0]
                dependency_child_tag = dependency_child.tag
                dependency_child_feature = dependency_child.data
                if pattern_layer==0:
                    #print (pattern_child_tag, dependency_child_tag)
                    if pattern_child_tag == dependency_child_tag:
                        count = count + 1
                if pattern_layer==1 and (pattern_child_tag.count("{+METHOD}") > 0 or pattern_child_tag.count("{+TASK}") > 0):
                    #pattern_child_tag_temp = pattern_child_tag.replace("{+METHOD}","").replace("{+TASK}","")
                    #if pattern_child_feature == dependency_child_feature and pattern_child_tag_temp==dependency_child_tag:
                    #print(pattern_child_tag, dependency_child_tag, dependency_child_feature, pattern_child_feature)
                    if pattern_child_feature == dependency_child_feature:
                            count = count + 1

    if category=="M":
        if count>=2:
            return True
        else:
            return False

    if category=="T":
        if count>=2:
            return True
        else:
            return False

def deleteSubtree(select_phrase_tree, category, adj_list = None, stop_list = None):
    '''

    :param select_phrase_tree: Entity tree to be pruned
    :param category: Method word or task word
    :param adj_list: List of unnecessary meaningless adjectives
    :return:
    '''
    #nmod:such_as
    # Method and TASK are different and need to be considered according to different situations
    # Here we also delete adjectives from words, which exist at the beginning or end

    #'M'
    need_del_phrase_M = ['mark',  'ref', 'advcl', 'nummod', 'acl', 'det', 'case', 'cc']
    #'T'
    need_del_phrase_T = ['mark',  'ref', 'advcl', 'nummod', 'acl', 'det', 'case', 'cc']

    need_del_phrase = ['acl',  'nmod:by', 'ref', 'advcl', 'nummod',  'acl:relcl', 'mwe', "nmod:than"]

    need_del_pattern = ['det', 'cc', 'cop', 'neg', 'advmod', 'mark', 'aux', 'dep', 'case', 'punct', 'amod']
    node_list = select_phrase_tree.all_nodes()
    length = len(node_list)
    root = select_phrase_tree.root

    #Find the smallest id and the largest id
    def delete_inter(select_phrase_tree):
        id_list = []
        node_list = select_phrase_tree.all_nodes()
        for node in node_list:
            node_id = node.identifier
            id_list.append(node_id)
        max_id = max(id_list)
        min_id = min(id_list)
        max_node = select_phrase_tree.get_node(max_id)
        min_node = select_phrase_tree.get_node(min_id)
        max_node_tag = max_node.tag
        max_node_feature = max_node.data
        min_node_tag = min_node.tag
        min_node_feature = min_node.data
        if max_node_tag=="," or max_node_tag=="." or max_node_tag==":" or max_node_tag=="?":
            if select_phrase_tree.contains(max_id):
                try:
                    select_phrase_tree.remove_node(max_id)
                except:
                    print('phrase cant delete')

        if min_node_tag=="," or min_node_tag=="." or min_node_tag==":" or min_node_tag=="?":
            if select_phrase_tree.contains(min_id):
                try:
                    select_phrase_tree.remove_node(min_id)
                except:
                    print('phrase cant delete')

        if max_id!=root:
                if category=="phrase_M":
                    if need_del_phrase_M.count(max_node_feature)>0:
                        if select_phrase_tree.contains(max_id):
                            try:
                                select_phrase_tree.remove_node(max_id)
                            except:
                                print ('phrase cant delete')
                if category=="phrase_T":
                    if need_del_phrase_T.count(max_node_feature)>0:

                        if select_phrase_tree.contains(max_id):
                            try:
                                select_phrase_tree.remove_node(max_id)
                            except:
                                print ('phrase cant delete')
        if min_id!=root:
                if category=="phrase_M":
                    if need_del_phrase_M.count(min_node_feature)>0 or adj_list.count(min_node_tag)>0 or stop_list.count(min_node_tag)>0:
                        if select_phrase_tree.contains(min_id):
                            try:
                                select_phrase_tree.remove_node(min_id)
                            except:
                                print ('phrase cant delete')
                if category=="phrase_T":
                    if need_del_phrase_T.count(min_node_feature)>0 or adj_list.count(min_node_tag)>0 or stop_list.count(min_node_tag)>0:
                        if select_phrase_tree.contains(min_id):
                            try:
                                select_phrase_tree.remove_node(min_id)
                            except:
                                print ('phrase cant delete')
        return select_phrase_tree

    for _ in range(0, length):
        if select_phrase_tree!=None and len(select_phrase_tree.all_nodes())>0:
            select_phrase_tree = delete_inter(select_phrase_tree)
        else:
            select_phrase_tree=None


    for node in node_list:
        node_id = node.identifier
        node_tag = node.tag
        node_feature = node.data

        if node_id != root:
            if category=="phrase_M" or category=="phrase_T":
                if need_del_phrase.count(node_feature)>0:
                    if node_tag!='and':
                        if select_phrase_tree.contains(node_id):
                            try:
                                select_phrase_tree.remove_node(node_id)
                            except:
                                print('phrase cant delete')
            if category=="pattern":
                if need_del_pattern.count(node_feature)>0:
                    if select_phrase_tree.contains(node_id):
                        try:
                            select_phrase_tree.remove_node(node_id)
                        except:
                            print('phrase cant delete')
    return select_phrase_tree

def deleteSameDepend(depend):
    alls = []
    need = []
    for item in depend:
        head=item[1]
        end=item[2]
        judge = 1
        if alls!=[]:
            for all in alls:
                if len(set(all).intersection(set([head, end])))==2:
                    judge = 0
                    break
            if judge==1:
                alls.append([head, end])
                need.append(item)
        else:
            alls.append([head, end])
            need.append(item)
    return need

def extractPattern(phrase_tree_layerOrder, dependency_sub_tree_layerOrder, phrase_tree, dependency_sub_tree, dependency_tree, category):
    '''   first step: judge phrase tree is a sub tree of dependency sub tree'''
    phrase_nodes = phrase_tree.all_nodes()
    nodes_quantity = len(phrase_nodes)
    count = 0
    hasSameTree = True

    for phrase_layer, phrase_values in phrase_tree_layerOrder.items():
        dependency_values = dependency_sub_tree_layerOrder[phrase_layer]
        phrase_values_len = len(phrase_values)
        count_sub = 0
        for phrase_value in phrase_values:
            phrase_child = phrase_value[0]
            phrase_head = phrase_value[1]
            phrase_child_tag = phrase_child.tag.replace("{+METHOD}","").replace("{+TASK}","")
            phrase_child_feature = phrase_child.data
            phrase_head_tag = phrase_head.tag.replace("{+METHOD}","").replace("{+TASK}","")
            for dependency_value in dependency_values:
                dependency_child = dependency_value[0]
                dependency_head = dependency_value[1]
                dependency_child_tag = dependency_child.tag.replace("{+METHOD}","").replace("{+TASK}","")
                dependency_child_feature = dependency_child.data
                dependency_head_tag = dependency_head.tag.replace("{+METHOD}","").replace("{+TASK}","")
                if phrase_layer>0:#根节点具有不同的处理
                    if (phrase_child_tag == dependency_child_tag
                          and phrase_child_feature == dependency_child_feature
                          and phrase_head_tag == dependency_head_tag
                    ):
                        count_sub = count_sub + 1
                        count = count + 1
                else:
                    if (phrase_child_tag == dependency_child_tag
                          ):
                        count_sub = count_sub + 1
                        count = count + 1
        if phrase_values_len != count_sub:
            hasSameTree = False
            break
    if nodes_quantity != count:
        hasSameTree=False

    if hasSameTree:# if same
        root_id = dependency_sub_tree.root
        dependency_sub_tree_root_tag = dependency_sub_tree.get_node(root_id).tag.replace("{+METHOD}","").replace("{+TASK}","")
        dependency_sub_tree_root_data = dependency_sub_tree.get_node(root_id).data
        if category=="M":
            dependency_tree.update_node(root_id, tag=dependency_sub_tree_root_tag+"{+METHOD}", data = dependency_sub_tree_root_data)
        else:
            dependency_tree.update_node(root_id, tag=dependency_sub_tree_root_tag +"{+TASK}", data = dependency_sub_tree_root_data)

        #one layer parent pattern
        parent_node = dependency_tree.parent(root_id)
        if parent_node!=None: #if exist parent node
            parents_childs_nodes = dependency_tree.children(parent_node.identifier)

            pattern_tree = Tree()
            pattern_tree.create_node(parent_node.tag, parent_node.identifier)
            for parents_childs_node in parents_childs_nodes:
                pattern_tree.create_node(parents_childs_node.tag, parents_childs_node.identifier,
                                         parent=parent_node.identifier, data=parents_childs_node.data)

            return pattern_tree
        else:
            return None

def findSeed(dependency_tree, dependency_tree_lemma, phrase_tree, phrase_tree_lemma, tokens, tokens_lemma, category, if_seed=False, use_lemma=False):
    '''
    step:
    1. judge if tokens in phrase_tree are all can be found in tokens list
    2. find the root and select the sub dependency_tree
    3. judge the similarity of phrase tree and dependency_tree
    4. use the root of phrase_tree to find the upper pattern
    using phrase to extract pattern
    :param dependency_tree:
    :param phrase_tree:
    :return:
    '''
    if use_lemma:
        has_phrase_word = containsToken(phrase_tree_lemma, tokens_lemma)
    else:
        has_phrase_word = containsToken(phrase_tree, tokens)

    select_pattern_trees = []
    if has_phrase_word:
        if use_lemma:
            if if_seed:
                dependency_sub_trees = getSeedTree(dependency_tree_lemma, phrase_tree_lemma)
            else:
                dependency_sub_trees = getSubTree(dependency_tree_lemma, phrase_tree_lemma)

            if dependency_sub_trees!=[]:
                for dependency_sub_tree in dependency_sub_trees:
                    'Traversing by level'
                    dependency_sub_tree_layerOrder = levelOrder(dependency_sub_tree)
                    phrase_tree_layerOrder = levelOrder(phrase_tree_lemma)
                    try:
                        select_pattern_tree = extractPattern(phrase_tree_layerOrder, dependency_sub_tree_layerOrder, phrase_tree_lemma, dependency_sub_tree, dependency_tree_lemma, category)
                    except:
                        select_pattern_tree=None

                    if select_pattern_tree!=None:
                        select_pattern_trees.append(select_pattern_tree)
        else:
            if if_seed:
                dependency_sub_trees = getSeedTree(dependency_tree, phrase_tree)
            else:
                dependency_sub_trees = getSubTree(dependency_tree, phrase_tree)
            if dependency_sub_trees != []:
                for dependency_sub_tree in dependency_sub_trees:
                    '{' \
                    '0: [[Node(tag=approach, identifier=7, data=None), Node(tag=approach, identifier=7, data=None)]], ' \
                    '1: [[Node(tag=of, identifier=4, data=case), Node(tag=approach, identifier=7, data=None)], ' \
                    '[Node(tag=the, identifier=5, data=det), Node(tag=approach, identifier=7, data=None)], ' \
                    '[Node(tag=POMDP, identifier=6, data=compound), Node(tag=approach, identifier=7, data=None)], ' \
                    '[Node(tag=system, identifier=11, data=nmod:to), Node(tag=approach, identifier=7, data=None)]]]}'
                    dependency_sub_tree_layerOrder = levelOrder(dependency_sub_tree)
                    phrase_tree_layerOrder = levelOrder(phrase_tree)
                    try:
                        select_pattern_tree = extractPattern(phrase_tree_layerOrder, dependency_sub_tree_layerOrder, phrase_tree,
                                                         dependency_sub_tree, dependency_tree, category)
                    except:
                        select_pattern_tree = None

                    if select_pattern_tree != None:
                        select_pattern_trees.append(select_pattern_tree)

    return select_pattern_trees

def useSemevalFindPattern():
    ann_label = {}
    id_name = {}
    with open("FES/train/ids_train") as fr:
        for line in fr:
            id, name = line.strip().split("\t")[0], line.strip().split("\t")[1]
            name = name.replace(".data",'')
            id_name[int(id)] = name
    with open("FES/train/train") as fr:
        for line in fr:
            id, label = line.strip().split("\t")[0], line.strip().split("\t")[2]
            id = int(id)-1
            name = id_name[id]
            ann_label[name] = label

    patterns_M_new = []
    patterns_T_new = []
    patterns_M_string = set()
    patterns_T_string = set()

    with open("FES/select_from_semeval_train.txt","r") as file:
        for line in file.readlines():
            sp = line.split("\t")
            filename = sp[0]
            file_ann = sp[1]
            ori_trees = []
            ori_sentences = []
            ori_sentences_lemma = []
            patterns_M = []
            patterns_T = []

            entity_pattern_M = []
            entity_pattern_T = []

            entities_M = []
            entities_T = []

            with open("../data/semeval/train_json/"+filename+".json", "r") as f_ori:
                for line in f_ori.readlines():
                    '''json process'''
                    json_line = json.loads(line)
                    sents = json_line['word']
                    if "1" in sents:
                        for sen_id, sentence in sents.items():
                            tokens = sentence['token']  # tokens word list
                            depend_tree = sentence['depend']  # dependency tree list (feature, start, end)
                            'delete nosense tree part'
                            depend_tree = deleteSameDepend(depend_tree) # delete nosense tree part
                            lemma = sentence['lemma']  # stemmed result  word list
                            ori_trees.append(depend_tree)
                            ori_sentences.append([s.lower() for s in tokens])
                            ori_sentences_lemma.append([s.lower() for s in lemma])

            method_type = ["Method",'Material','Metric']
            task_type = ['Task']
            json_dict = json.loads(file_ann)
            sens = json_dict['sentences']
            ner_res = json_dict['ner']
            all_tokens = []
            sen_length = []
            entity_dict_Ms = {}
            entity_dict_Ts = {}
            for sen in sens:
                all_tokens.extend(sen)
                sen_length.append(len(sen))
            for sen_id, ners in enumerate(ner_res):
                phrase_id = 0
                for ner in ners:
                        start, end, type = ner[0], ner[1], ner[2]
                        if filename + "_" + str(sen_id) in ann_label and ann_label[filename + "_" + str(sen_id)] == "M":
                            last_length = sum(sen_length[:sen_id])
                            this_start = start-last_length
                            this_end = end - last_length
                            phrase_dict = {}
                            for word_id in range(this_start, this_end+1):
                                phrase_dict[word_id] = sens[sen_id][word_id]
                            if sen_id not in entity_dict_Ms:
                                entity_dict_Ms[sen_id] = {phrase_id:phrase_dict}
                            else:
                                entity_dict_Ms[sen_id][phrase_id] = phrase_dict
                            phrase_id +=1
                        if filename + "_" + str(sen_id) in ann_label and ann_label[filename + "_" + str(sen_id)] == "T":
                            last_length = sum(sen_length[:sen_id])
                            this_start = start - last_length
                            this_end = end - last_length
                            phrase_dict = {}
                            for word_id in range(this_start, this_end + 1):
                                phrase_dict[word_id] = sens[sen_id][word_id]
                            if sen_id not in entity_dict_Ts:
                                entity_dict_Ts[sen_id] = {phrase_id: phrase_dict}
                            else:
                                entity_dict_Ts[sen_id][phrase_id] = phrase_dict
                            phrase_id += 1

                print(entity_dict_Ms)
                print(entity_dict_Ts)

            for sen_id, (ori_sentence, ori_sentence_lemma, ori_tree) in enumerate(zip(sens, ori_sentences_lemma, ori_trees)):
                def constructPhraseTree(entity_dict, ori_sentence, ori_sentence_lemma, ori_tree):
                    '''
                            return: list
                            The function of constructing the tree of entity words
                            1. Judge the entity_dict_T or entity_dict_M dictionary is empty
                            2. According to the position of words, the dependency syntax tree representation of entity words is obtained
                            3. Construct the tree structure of entity words
                            4. Save the tree into sub_trees_M and sub_trees_T
                            Return: List form. Tree in the list
                     '''
                    sub_trees, sub_trees_lemma = [], []
                    if entity_dict != {}:
                        for idx, entity in entity_dict.items():
                            all_posi = []  # The position of each word in the entity
                            ori_sub_tree = []
                            for posi in entity.keys():
                                all_posi.append(posi)
                            for depen in ori_tree:
                                head = int(depen[1].split("-")[0])
                                child = int(depen[2].split("-")[0])
                                if len(all_posi) > 1:
                                    if head in all_posi and child in all_posi:
                                        ori_sub_tree.append(depen)
                                else:
                                    if head in all_posi:
                                        # An entity consisting of only one word
                                        ori_sub_tree.append(depen)
                                        break
                            'May form multiple trees'
                            sub_tree, sub_tree_lemma = constructDependencyTree(ori_sub_tree, ori_sentence, ori_sentence_lemma,
                                                                               len(all_posi))
                            for sub, sub_lemma in zip(sub_tree, sub_tree_lemma):
                                sub_trees.append(sub)
                                sub_trees_lemma.append(sub_lemma)
                    return sub_trees, sub_trees_lemma

                '''Build a tree of entity words. Reference the internal function structPhraseTree'''
                sub_trees_M, sub_trees_lemma_M = [], []
                sub_trees_T, sub_trees_lemma_T = [], []
                if sen_id in entity_dict_Ms and entity_dict_Ms[sen_id] != {}:
                    sub_trees_M, sub_trees_lemma_M = constructPhraseTree(entity_dict_Ms[sen_id], ori_sentence, ori_sentence_lemma, ori_tree)
                if sen_id in entity_dict_Ts and entity_dict_Ts[sen_id] != {}:
                    sub_trees_T, sub_trees_lemma_T = constructPhraseTree(entity_dict_Ts[sen_id], ori_sentence, ori_sentence_lemma, ori_tree)

                '''
                Obtain template through entity subtree and sentence tree
                '''
                if sub_trees_M != []:
                    for sub_tree_M, sub_tree_lemma_M in zip(sub_trees_M, sub_trees_lemma_M):
                        if containsTokenAll(sub_tree_lemma_M, ori_sentence_lemma):
                            trees, trees_lemma = constructDependencyTree(ori_tree, ori_sentence, ori_sentence_lemma,
                                                                         len(ori_sentence))
                            for tree, tree_lemma in zip(trees, trees_lemma):
                                select_pattern_trees = findSeed(tree, tree_lemma, sub_tree_M, sub_tree_lemma_M, ori_sentence,
                                                                ori_sentence_lemma, category='M', if_seed=True, use_lemma=True)
                                if select_pattern_trees != []:
                                    for select_pattern_tree in select_pattern_trees:
                                        patterns_M.append(select_pattern_tree)
                                    entity_pattern_M.append(select_pattern_trees)
                                    entities_M.append(sub_tree_M)

                if sub_trees_T != []:
                    for sub_tree_T, sub_tree_lemma_T in zip(sub_trees_T, sub_trees_lemma_T):
                        if containsTokenAll(sub_tree_lemma_T, ori_sentence_lemma):
                            trees, trees_lemma = constructDependencyTree(ori_tree, ori_sentence, ori_sentence_lemma,
                                                                         len(ori_sentence))
                            for tree, tree_lemma in zip(trees, trees_lemma):
                                select_pattern_trees = findSeed(tree, tree_lemma, sub_tree_T, sub_tree_lemma_T, ori_sentence,
                                                                ori_sentence_lemma, category='T', if_seed=True, use_lemma=True)
                                if select_pattern_trees != []:
                                    for select_pattern_tree in select_pattern_trees:
                                        patterns_T.append(select_pattern_tree)
                                    entity_pattern_T.append(select_pattern_trees)
                                    entities_T.append(sub_tree_T)
                #print (len(patterns_M))
                #print(len(patterns_T))
                for M in patterns_M:
                    M = deleteSubtree(M, category='pattern')
                    if M!=None:
                        method = str(M.to_dict(with_data=True))
                        if method not in patterns_M_string:
                            patterns_M_string.add(method)
                            patterns_M_new.append(M)

                for T in patterns_T:
                    T = deleteSubtree(T, category='pattern')
                    if T!=None:
                        task = str(T.to_dict(with_data=True))
                        if task not in patterns_T_string:
                            patterns_T_string.add(task)
                            patterns_T_new.append(T)

        print('patterns_T', len(patterns_T))
        print('patterns_M', len(patterns_M))

        'Store all found seed pattern trees'
        with open("FES/seed_patternz_M", "w") as fw_seed_pattern_M:
            for method in patterns_M_new:
                method = method.to_dict(with_data=True)
                fw_seed_pattern_M.write(json.dumps(method) + '\n')
        fw_seed_pattern_M.close()

        with open("FES/seed_patternz_T", "w") as fw_seed_pattern_T:
            for task in patterns_T_new:
                task = task.to_dict(with_data=True)
                fw_seed_pattern_T.write(json.dumps(task) + '\n')
        fw_seed_pattern_T.close()

if __name__ == '__main__':
    'using denpendency tree find pattern'
    path_log = 'all.log'
    useSemevalFindPattern()

