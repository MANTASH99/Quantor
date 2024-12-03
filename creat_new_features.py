import pandas as pd
import numpy as np


data = pd.read_csv('ICESING.csv', delimiter='\t')

print(data.head())
data['word_S'] = data['n_word'] / data['n_sent']
data['lexical_density'] = data['ld'] / data['n_word']
data['nn_W'] = data['nn'] / data['n_word']
data['np_W'] = data['np'] / data['n_word']
data['nominal_W'] = data['nom'] / data['n_word']
data['neoclass_W'] = data['neo'] / data['n_word']
data['poss_pronoun_W'] = data['poss'] / data['n_word']
data['pronoun_all_W'] = data['pronoun'] / data['n_word']
data['p1_perspron_P'] = data['p1'] / data['pronoun']
data['p2_perspron_P'] = data['p2'] / data['pronoun']
data['p3_perspron_P'] = data['p3'] / data['pronoun']
data['it_P'] = data['pit'] / data['pronoun']
data['pospers1_W'] = data['pospers1'] / data['n_word']
data['pospers2_W'] = data['pospers2'] / data['n_word']
data['pospers3_W'] = data['pospers3'] / data['n_word']
data['adj_W'] = data['adj'] / data['n_word']
data['atadj_W'] = data['atadj'] / data['n_word']
data['predadj_W'] = (data['adj'] - data['atadj']) / data['n_word']
data['prep_W'] = data['prep'] / data['n_word']
data['finite_S'] = data['fin'] / data['n_sent']
data['past_tense_F'] = data['past'] / data['fin']
data['will_F'] = data['will'] / data['fin']
data['modal_verb_V'] = data['vm'] / data['v']
data['verb_W'] = data['v'] / data['n_word']
data['infinitive_F'] = data['inf'] / data['fin']
data['passive_F'] = data['pass'] / data['fin']
data['coordination_F'] = data['coord'] / data['fin']
data['subordination_F'] = data['subord'] / data['fin']
data['interrogative_S'] = data['interr'] / data['n_sent']
data['imperative_S'] = data['imper'] / data['n_sent']
data['title_W'] = data['title'] / data['n_word']
data['salutation_S'] = data['salutgreet'] / data['n_sent']
data['place_adv_W'] = data['rl'] / data['n_word']
data['time_adv_W'] = data['rt'] / data['n_word']
data['nom_initial_S'] = data['nptheme'] / data['n_sent']
data['prep_initial_S'] = data['pptheme'] / data['n_sent']
data['adv_initial_S'] = data['advtheme'] / data['n_sent']
data['text_initial_S'] = data['cctheme'] / data['n_sent']
data['wh_initial_S'] = data['whtheme'] / data['n_sent']
data['disc_initial_S'] = data['disctheme'] / data['n_sent']
data['nonfin_initial_S'] = data['totheme'] / data['n_sent']
data['subord_initial_S'] = data['subordtheme'] / data['n_sent']
data['verb_initial_S'] = data['verbtheme'] / data['n_sent']

# Save the updated DataFrame back to a CSV file
data.to_csv('updated_file.csv', index=False)
