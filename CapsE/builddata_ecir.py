import numpy as np
def read_from_id(filename='../data/WN18RR/entity2id.txt'):
    entity2id = {}
    id2entity = {}
    with open(filename) as f:
        for line in f:
            if len(line.strip().split()) > 1:
                tmp = line.strip().split()
                entity2id[tmp[0]] = int(tmp[1])
                id2entity[int(tmp[1])] = tmp[0]
    return entity2id, id2entity

def assignEmbeddings(lstent, word_indexes, embedding_dim=200):
    lstEmbedUser = np.empty([len(word_indexes), embedding_dim]).astype(np.float32)
    for word in word_indexes:
        _ind = word_indexes[word]
        lstEmbedUser[_ind] = lstent[word]
    return lstEmbedUser


def init_dataset_ecir(entinit):
    lstent = {}
    with open(entinit) as f:
        for line in f:
            lstval = line.strip().split()
            tmp = [float(val) for val in lstval[1:]]
            lstent[lstval[0]] = tmp
    return lstent

def init_norm_Vector(relinit, entinit, embedding_size):
    lstent = []
    lstrel = []
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstent.append(tmp)
    assert embedding_size % len(lstent[0]) == 0
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)


def getID(folder='data/WN18RR/'):
    lstEnts = {}
    lstRels = {}
    with open(folder + 'train.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)

    with open(folder + 'valid.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)

    with open(folder + 'test.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)

    wri = open(folder + 'entity2id.txt', 'w')
    for entity in lstEnts:
        wri.write(entity + '\t' + str(lstEnts[entity]))
        wri.write('\n')
    wri.close()

    wri = open(folder + 'relation2id.txt', 'w')
    for entity in lstRels:
        wri.write(entity + '\t' + str(lstRels[entity]))
        wri.write('\n')
    wri.close()


def parse_line(line):
    line = line.strip().split()
    sub = line[0]
    rel = line[1]
    obj = line[2]
    val = [1]
    if len(line) > 3:
        if line[3] == '-1':
            val = [-1]
    return sub, obj, rel, val

def load_triples_from_txt(filename, words_indexes=None, parse_line=parse_line):
    """
    Take a list of file names and build the corresponding dictionnary of triples
    """
    if words_indexes == None:
        words_indexes = dict()
        entities = set()
        next_ent = 0
    else:
        entities = set(words_indexes)
        next_ent = max(words_indexes.values()) + 1

    data = dict()

    with open(filename) as f:
        lines = f.readlines()

    for _, line in enumerate(lines):
        sub, obj, rel, val = parse_line(line)

        if sub in entities:
            sub_ind = words_indexes[sub]
        else:
            sub_ind = next_ent
            next_ent += 1
            words_indexes[sub] = sub_ind
            entities.add(sub)

        if rel in entities:
            rel_ind = words_indexes[rel]
        else:
            rel_ind = next_ent
            next_ent += 1
            words_indexes[rel] = rel_ind
            entities.add(rel)

        if obj in entities:
            obj_ind = words_indexes[obj]
        else:
            obj_ind = next_ent
            next_ent += 1
            words_indexes[obj] = obj_ind
            entities.add(obj)

        data[(sub_ind, rel_ind, obj_ind)] = val

    indexes_words = {}
    for tmpkey in words_indexes:
        indexes_words[words_indexes[tmpkey]] = tmpkey

    return data, words_indexes, indexes_words


def build_data(name='WN18', path='../../CNNGraph/data'):
    folder = path + '/' + name + '/'

    train_triples, words_indexes, _ = load_triples_from_txt(folder + 'train.txt', parse_line=parse_line)

    valid_triples, words_indexes, _ = load_triples_from_txt(folder + 'valid.txt',
                                                            words_indexes=words_indexes, parse_line=parse_line)

    test_triples, words_indexes, indexes_words = load_triples_from_txt(folder + 'test.txt',
                                                                       words_indexes=words_indexes,
                                                                       parse_line=parse_line)

    entity2id, id2entity = read_from_id(folder + '/entity2id.txt')
    relation2id, id2relation = read_from_id(folder + '/relation2id.txt')
    left_entity = {}
    right_entity = {}

    with open(folder + 'train.txt') as f:
        lines = f.readlines()
    for _, line in enumerate(lines):
        head, tail, rel, val = parse_line(line)
        # count the number of occurrences for each (heal, rel)
        if relation2id[rel] not in left_entity:
            left_entity[relation2id[rel]] = {}
        if entity2id[head] not in left_entity[relation2id[rel]]:
            left_entity[relation2id[rel]][entity2id[head]] = 0
        left_entity[relation2id[rel]][entity2id[head]] += 1
        # count the number of occurrences for each (rel, tail)
        if relation2id[rel] not in right_entity:
            right_entity[relation2id[rel]] = {}
        if entity2id[tail] not in right_entity[relation2id[rel]]:
            right_entity[relation2id[rel]][entity2id[tail]] = 0
        right_entity[relation2id[rel]][entity2id[tail]] += 1

    left_avg = {}
    for i in range(len(relation2id)):
        left_avg[i] = sum(left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_avg = {}
    for i in range(len(relation2id)):
        right_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_avg[i] / (right_avg[i] + left_avg[i])

    return train_triples, valid_triples, test_triples, words_indexes, indexes_words, headTailSelector, entity2id, id2entity, relation2id, id2relation

def dic_of_chars(words_indexes):
    lstChars = {}
    for word in words_indexes:
        for char in word:
            if char not in lstChars:
                lstChars[char] = len(lstChars)
    lstChars['unk'] = len(lstChars)
    return lstChars


def convert_to_seq_chars(x_batch, lstChars, indexes_words):
    lst = []
    for [tmpH, tmpR, tmpT] in x_batch:
        wH = [lstChars[tmp] for tmp in indexes_words[tmpH]]
        wR = [lstChars[tmp] for tmp in indexes_words[tmpR]]
        wT = [lstChars[tmp] for tmp in indexes_words[tmpT]]
        lst.append([wH, wR, wT])
    return lst

def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):
    sequence_padded, sequence_length = [], []
    max_length_word = max([max(map(lambda x: len(x), seq))
                           for seq in sequences])
    for seq in sequences:
        # all words are same length now
        sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]

    max_length_sentence = max(map(lambda x: len(x), sequences))
    sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
    sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    return np.array(sequence_padded).astype(np.int32), np.array(sequence_length).astype(np.int32)


def parse_line_ecir(line, query, user):
    line = line.strip().split()
    if len(line) == 5:
        sub = line[2]
        rel = line[3]
        obj = line[4]
        val = [1]
        rank = int(line[1].split('-')[1])
        return sub, rel, obj, val, rank, 1
    elif len(line) == 3:
        rank = int(line[1].split('-')[1])
        sub = query
        rel = user
        obj = line[2]
        val = [-1]
        return sub, rel, obj, val, rank, -1
    else:
        return None, None, None, None, None, 0


def load_triples_from_txt_ecir(filename, query_indexes=None, user_indexes=None, doc_indexes=None):
    """
    Take a list of file names and build the corresponding dictionnary of triples
    """
    if user_indexes == None:
        user_indexes = dict()
        user_entities = set()
        user_next_ent = 0
    else:
        user_entities = set(user_indexes)
        user_next_ent = max(user_indexes.values()) + 1

    if doc_indexes == None:
        doc_indexes = dict()
        doc_entities = set()
        doc_next_ent = 0
    else:
        doc_entities = set(doc_indexes)
        doc_next_ent = max(doc_indexes.values()) + 1

    if query_indexes == None:
        query_indexes = dict()
        query_entities = set()
        query_next_ent = 0
    else:
        query_entities = set(query_indexes)
        query_next_ent = max(query_indexes.values()) + 1

    lsttriples = []
    lstranks = []
    lstvals = []
    lsttriple = []
    lstrank = []
    lstval = []
    with open(filename) as f:
        lines = f.readlines()

    query = ''
    user = ''
    for _, line in enumerate(lines):
        query, user, doc, val, rank, _star = parse_line_ecir(line, query, user)
        #print(query, user, doc, val, rank)
        if rank == None:
            continue

        if query in query_entities:
            query_ind = query_indexes[query]
        else:
            query_ind = query_next_ent
            query_next_ent += 1
            query_indexes[query] = query_ind
            query_entities.add(query)

        if user in user_entities:
            user_ind = user_indexes[user]
        else:
            user_ind = user_next_ent
            user_next_ent += 1
            user_indexes[user] = user_ind
            user_entities.add(user)

        if doc in doc_entities:
            doc_ind = doc_indexes[doc]
        else:
            doc_ind = doc_next_ent
            doc_next_ent += 1
            doc_indexes[doc] = doc_ind
            doc_entities.add(doc)

        if _star == 1 and len(lsttriple) > 1:
            lsttriple = np.array(lsttriple)
            lstrank = np.array(lstrank)
            lstval = np.array(lstval)

            lsttriples.append(lsttriple)
            lstranks.append(lstrank)
            lstvals.append(lstval)

            lsttriple = []
            lstrank = []
            lstval = []

        lsttriple.append(np.array([query_ind, user_ind, doc_ind]))
        lstrank.append(rank)
        lstval.append(val)

    lsttriple = np.array(lsttriple)
    lstrank = np.array(lstrank)
    lstval = np.array(lstval)

    lsttriples.append(lsttriple)
    lstranks.append(lstrank)
    lstvals.append(lstval)

    lsttriples = np.array(lsttriples)
    lstranks = np.array(lstranks)
    lstvals = np.array(lstvals)

    return lsttriples, lstranks, lstvals, query_indexes, user_indexes, doc_indexes

def build_data_ecir(name='SEARCH17', path='./data'):
    folder = path + '/' + name + '/'

    train_triples, train_rank_triples, train_val_triples, query_indexes, user_indexes, doc_indexes \
        = load_triples_from_txt_ecir(folder + 'sample_train.200.txt')
    #print(len(query_indexes), len(user_indexes), len(doc_indexes))

    valid_triples, valid_rank_triples, valid_val_triples, query_indexes, user_indexes, doc_indexes \
        = load_triples_from_txt_ecir(folder + 'sample_dev.200.txt',
                user_indexes=user_indexes, query_indexes=query_indexes, doc_indexes=doc_indexes)
    #print(len(query_indexes), len(user_indexes), len(doc_indexes))

    test_triples, test_rank_triples, test_val_triples, query_indexes, user_indexes, doc_indexes \
        = load_triples_from_txt_ecir(folder + 'sample_test.200.txt',
                user_indexes=user_indexes, query_indexes=query_indexes, doc_indexes=doc_indexes)
    #print(len(query_indexes), len(user_indexes), len(doc_indexes))

    indexes_user = {}
    for tmp in user_indexes:
        indexes_user[user_indexes[tmp]] = tmp

    indexes_query = {}
    for tmp in query_indexes:
        indexes_query[query_indexes[tmp]] = tmp

    indexes_doc = {}
    for tmp in doc_indexes:
        indexes_doc[doc_indexes[tmp]] = tmp


    return train_triples, train_rank_triples, train_val_triples, valid_triples, valid_rank_triples, valid_val_triples, \
            test_triples, test_rank_triples, test_val_triples, query_indexes, user_indexes, doc_indexes, \
            indexes_query, indexes_user, indexes_doc

class Batch_Loader_ecir(object):
    def __init__(self, train_triples, train_val_triples, batch_size=100):

        self.train_triples = train_triples
        self.train_val_triples = train_val_triples
        self.batch_size = batch_size

    def __call__(self):

        idxs = np.random.randint(0, len(self.train_val_triples), self.batch_size)
        self.new_triples_indexes = np.concatenate(self.train_triples[idxs])
        self.new_triples_values = np.concatenate(self.train_val_triples[idxs], axis=0)

        while len(self.new_triples_indexes) < self.batch_size * 10:
            self.new_triples_indexes = np.append(self.new_triples_indexes, self.new_triples_indexes, axis=0)
            self.new_triples_values = np.append(self.new_triples_values, self.new_triples_values, axis=0)

        self.new_triples_indexes = np.append(self.new_triples_indexes, self.new_triples_indexes[:(self.batch_size * 20 - self.new_triples_values.shape[0])], axis=0)
        self.new_triples_values = np.append(self.new_triples_values, self.new_triples_values[:(self.batch_size * 20 - self.new_triples_values.shape[0])], axis=0)

        return self.new_triples_indexes.astype(np.int32), self.new_triples_values.astype(np.float32)

def computeMRR(lstRanks):
    rr = 0.0
    for tmp in lstRanks:
        rr += 1.0/ tmp[0]
    return rr / len(lstRanks)

def computeP1(lstRanks):
    p1 = 0.0
    for tmp in lstRanks:
        if tmp[0] == 1:
            p1 += 1
    return p1 / len(lstRanks)

