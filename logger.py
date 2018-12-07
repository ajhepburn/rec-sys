from analysis import Analysis, Doc2VecAnalysis
from utils import load_model

class Logger:
    def __init__(self, model_name):
        self.log_path = './log/'
        self.model_path = './models/'
        self.model_name = model_name
        self.model = load_model(self.model_path, self.model_name, type)

class AnalysisLogger(Logger):
    def write_stats_to_file(self):
        with open(self.log_path+"log_"+self.model_name+".txt") as fp:
            title = "Model: "+self.model_name
            print("\n"+title+"\n"+("-"*(len(title)+1)))
            fp.write('Vocab Size:',self.model.get_vocab_size(), end="\n")
            fp.write('Vocab Size:',self.model.get_vocab_size(), end="\n")
            fp.write('Document Count:',self.model.get_number_of_docs(), end="\n")
            fp.write('Apple:',self.model.most_similar_words('apple'), end="\n")
            fp.write('Google:',self.model.most_similar_words('google'), end="\n")
            fp.write('Tesla:',self.model.most_similar_words('tesla'), end="\n")
            fp.write('King + Woman - Man =',self.model.subtract_from_vectors('king','woman','man'), end="\n")
            fp.write('{0} + {1} - {2} ='.format('Paris','England','London'),self.model.subtract_from_vectors('paris','england','london'))