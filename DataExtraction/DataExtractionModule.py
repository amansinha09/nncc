class DOCUMENT_DATA_EXTRACTOR():
    
    def __init__(self,PATH,name,tagging = "BIO2",default_open_saved_file_if_exists = True,default_save_read_file = True):
        self.name = name
        globals()[self.name] = {}
        self.PATH = PATH
        self.default_open_saved_file_if_exists = default_open_saved_file_if_exists
        self.default_save_read_file = default_save_read_file
        assert(tagging in ["BIO2"]),str(tagging) + " tagging scheme is currently not available"
        self.tagging = tagging
    
    def extract(self,file_no,training,save_read_file = None,open_saved_file_if_exists = None):
        
        path = self.PATH + file_no
        
        if open_saved_file_if_exists or (open_saved_file_if_exists == None and self.default_open_saved_file_if_exists):
            if path in globals()[self.name]:
                return globals()[self.name][path]
                
        with open(path + '.txt') as txt:
            raw = txt.read()

        sentences = wordpunct_tokenize(raw)
        spans = []

        span = 0
        for i in range(len(sentences)):
            if i == len(sentences):
                break
            while (raw[span:].find(sentences[i]) == -1):
                sentences.remove(sentences[i])
                if i == len(sentences):
                    break
            span += raw[span:].find(sentences[i])
            spans.append((span,span + len(sentences[i])))

        assert(len(sentences) == len(spans))

        if training == True:
            df = pd.read_csv( path + '.ann', sep='^([^\s]*)\s', engine='python', header=None).drop(0, axis=1)

            text_annotation = {}
            entities = {}
            attributes = {}

            for i in range(len(df)):
                if df.iloc[i,0].startswith("T"):
                    spltobj = df.iloc[i,1].split('\t')[0].split(chr(32))
                    text_annotation[df.iloc[i,0]] = (spltobj[1],spltobj[-1])
                elif df.iloc[i,0].startswith("E"):
                    entities[df.iloc[i,0]] = df.iloc[i,1].split(":")
                elif df.iloc[i,0].startswith("A"):
                    attributes[df.iloc[i,0]] = df.iloc[i,1].split(chr(32))

            txtspans = sorted([(int(text_annotation[x][0]),int(text_annotation[x][1]),x) for x in text_annotation])[::-1]

            text_annotation_indices = {}

            for i in text_annotation:
                text_annotation_indices[i] = []

            ptr = 0

            while len(txtspans)>0:

                while len(txtspans)>0 and txtspans[-1][1] <= spans[ptr][0]:
                    txtspans.pop()

                if len(txtspans) == 0:
                    break

                if spans[ptr][0] >= txtspans[-1][0] and spans[ptr][1] <= txtspans[-1][1]:

                    if text_annotation_indices[txtspans[-1][2]].__len__() == 0:
                        assert(txtspans[-1][0] == spans[ptr][0])

                    text_annotation_indices[txtspans[-1][2]].append(ptr)
                ptr += 1

            assert(len(txtspans) == 0 and ptr < len(spans))    
            
            if tagging == "BIO2":

                labels = np.zeros(len(spans),dtype = np.int8)

                for i in entities:
                    if entities[i][0] == "Disposition":
                        labels[text_annotation_indices[entities[i][1]]] = 1
                        for k in text_annotation_indices[entities[i][1]][1:]:
                            labels[k] = 2
                    elif entities[i][0] == "NoDisposition":
                        labels[text_annotation_indices[entities[i][1]]] = 3
                        for k in text_annotation_indices[entities[i][1]][1:]:
                            labels[k] = 4
                    elif entities[i][0] == "Undetermined":
                        labels[text_annotation_indices[entities[i][1]]] = 5
                        for k in text_annotation_indices[entities[i][1]][1:]:
                            labels[k] = 6
                label_tags = ["O","B-Disposition","I-Disposition","B-NoDisposition","I-NoDisposition","B-Undetermined","I-Undetermined"]     

            attribute_labels = np.zeros((len(spans),5),dtype = np.int8)  
            attribute_tags = ["Action","Actor","Certainty","Temporality","Negation"]
            Action_tags = ['Start','Stop','Increase',"Decrease", "OtherChange", "UniqueDose", "Unknown"]
            Actor_tags = ["Physician", "Patient", "Unknown"]
            Certainty_tags = ["Certain", 'Hypothetical', "Conditional", "Unknown"]
            Temporality_tags = ["Past", "Present", "Future", "Unknown"]
            Negation_tags = ["Negated", "NotNegated"]
            attr_types_tags = [Action_tags,Actor_tags,Certainty_tags,Temporality_tags,Negation_tags]

            for m in attributes:
                temp = attributes[m]
                for k in text_annotation_indices[entities[temp[1]][1]]:
                    attribute_labels[k][attribute_tags.index(temp[0])] = attr_types_tags[attribute_tags.index(temp[0])].index(temp[2])
        retval = {"sentences":sentences,
                  "labels":labels,
                  "attribute_labels":attribute_labels,
                  "tagging_scheme":{"label_tags":label_tags,"attribute_labels":attribute_labels,"attr_types_tags":attr_types_tags}}
        if save_read_file or (save_read_file == None and self.default_save_read_file):
            globals()[self.name][path] = retval
        return retval