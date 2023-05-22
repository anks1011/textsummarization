import numpy as np
import matplotlib
from Cython import inline
%matplotlib inline
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions
position1 = ctrl.Antecedent(np.arange(0, 100, 10), 'position1')
cos_similarity = ctrl.Antecedent(np.arange(0, 100, 10), 'cos_similarity')
bitokens = ctrl.Antecedent(np.arange(0, 100, 10), 'bitokens')
tritokens = ctrl.Antecedent(np.arange(0, 100, 10), 'tritokens')
propernoun = ctrl.Antecedent(np.arange(0, 100, 10), 'propernoun')
sentencelength = ctrl.Antecedent(np.arange(0, 100, 10), 'sentencelength')
numtokens = ctrl.Antecedent(np.arange(0, 100, 10), 'numtokens')
keywords = ctrl.Antecedent(np.arange(0, 10, 1), 'keywords')
tf_isf = ctrl.Antecedent(np.arange(0, 100, 10), 'tf_isf')


senten = ctrl.Consequent(np.arange(0, 100, 10), 'senten')

position1.automf(3)
cos_similarity.automf(3)
bitokens.automf(3)
tritokens.automf(3)
propernoun.automf(3)
sentencelength.automf(3)
numtokens.automf(3)
keywords.automf(3)
tf_isf.automf(3)


senten['bad'] = fuzz.trimf(senten.universe, [0, 0, 50])
senten['avg'] = fuzz.trimf(senten.universe, [0, 50, 100])
senten['good'] = fuzz.trimf(senten.universe, [50, 100, 100])

rule1 = ctrl.Rule(position1['good'] & sentencelength['good'] & propernoun['good'] &numtokens['good'], senten['good'])
rule2 = ctrl.Rule(position1['poor'] & sentencelength['poor'] & numtokens['poor'], senten['bad'])
rule3 = ctrl.Rule(propernoun['poor'] & keywords['average'], senten['bad'])
rule4 = ctrl.Rule(cos_similarity['good'], senten['good'])
rule5 = ctrl.Rule(bitokens['good'] & tritokens['good'] & numtokens['average'] | tf_isf['average'], senten['avg'])

sent_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5])
Sent = ctrl.ControlSystemSimulation(sent_ctrl)
fuzzemptyarr= np.empty((20,1,2), dtype=object)
t2=0
summary2=[]
for s in range(len(sentences)):
    Sent.input['position1'] = int(position[s])
    Sent.input['cos_similarity'] = int(cosine_similarity[s])
    Sent.input['bitokens'] = int(bi_tokens[s])
    Sent.input['tritokens'] = int(tri_tokens[s])
    Sent.input['tf_isf'] = int(tfisfvec[s])
    Sent.input['keywords'] = int(thematic_number[s])
    Sent.input['propernoun'] = int(pnounscore[s])
    Sent.input['sentencelength'] = int(sent_length[s])
    Sent.input['numtokens'] = int(numeric_token[s])
#Sent.input['service'] = 2
    Sent.compute()
    if Sent.output['senten'] > 50:
        summary2.append((sentences[s]))
        fuzzemptyarr[t2][0][0] = sentences[s]
        fuzzemptyarr[t2][0][1] = s
        t2+=1
fuzzarray = np.empty((len(summary2),1,2),dtype=object)
for i in range(len(summary2)):
    fuzzarray[i][0][0] = fuzzemptyarr[i][0][0]
    fuzzarray[i][0][1] = fuzzemptyarr[i][0][1]

fuzzarray=fuzzarray[1:]
print("Fuzzy logic summary \n\n",summary2)
