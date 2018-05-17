import msgpack
from keras.preprocessing.sequence import pad_sequences
import torch.optim as optim
from machine_comprehension import MachineComprehension


def train(passages,questions,answers,embeddings,c_maxlen,q_maxlen):
   
    mc = MachineComprehension(c_maxlen,q_maxlen,embeddings)
    optimizer = optim.RMSprop(mc.parameters())
    loss_start = torch.nn.NLLLoss()
    loss_end = torch.nn.NLLLoss()

    n_samples = len(passages)
    epochs = 100
    train = True
    for i in range(epochs):
        total_loss = 0
        for count,passage in enumerate(passages):
            c_var = Variable(torch.from_numpy(passage).type(torch.LongTensor))
            q_var = Variable(torch.from_numpy(questions[count]).type(torch.LongTensor))

            mc.passage_encoder.hidden_state = mc.passage_encoder.init_hidden()
            mc.question_encoder.hidden_state = mc.question_encoder.init_hidden()
            start_pred,end_pred = mc(c_var,q_var,train)
            loss = loss_start(start_pred.type(torch.DoubleTensor).squeeze(2),Variable(torch.from_numpy(np.array([answers[count][0]])))) + loss_end(end_pred.type(torch.DoubleTensor).squeeze(2),Variable(torch.from_numpy(np.array([answers[count][1]]))))
            total_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Total loss:",total_loss/n_samples)
    
    return mc
    
    
def predict(model,test_c,test_q,max_answer_span):
    s,e = mc(Variable(torch.from_numpy(test_c).type(torch.LongTensor)),Variable(torch.from_numpy(test_q).type(torch.LongTensor)),train=False)
    vector_outer_product = torch.ger(s.squeeze(0).squeeze(1),e.squeeze(0).squeeze(1))
    upper_triangular_values = vector_outer_product.triu()
    lower_triangular_values = upper_triangular_values.tril(max_answer_span - 1)
    all_scores = lower_triangular_values.numpy()
    max_score_idx = np.unravel_index(np.argmax(all_scores),all_scores.shape)
    max_start,max_end = max_score_idx[0],max_score_idx[1]
    return max_start,max_end
    

    
def main():
    with open(../"DrQA/SQuAD/sample.msgpack",'rb') as reader:
        sample = msgpack.unpack(reader,encoding='utf8')
    with open("../DrQA/SQuAD/meta.msgpack",'rb') as reader:
        meta = msgpack.unpack(reader,encoding='utf8')

    word2id = {word:i for i,word in enumerate(meta['vocab'])}
    id2word = {v:k for k,v in word2id.items()}

    passages = [context[1] for context in sample['train']]
    questions = [question[5] for question in sample['train']]
    answers = [[answer[-2],answer[-1]]for answer in sample['train']]


    passages = pad_sequences(passages,padding='post')
    questions = pad_sequences(questions,padding='post')

    embeddings  = meta['embedding']
    c_maxlen = max([len(i) for i in passages])
    q_maxlen = max([len(i) for i in questions])
    
    model = train(passages,questions,answers,embeddings,c_maxlen,q_maxlen)
    test_c = passages[1]
    test_q = questions[1]
    predict(model,test_c,test_q)
    

__main__()
