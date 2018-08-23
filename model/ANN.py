"""
This ANN model is inference layer of BNN network
"""

import tensorflow as tf 

folder_to_save_result = 'results/mem/5minutes/'
vector_state_file = folder_to_save_result + 'vector_presentation' + str(self.sliding_encoder) + '-' + str(self.sliding_decoder) + '-' + str(self.sliding_inference) + '-' + str(self.batch_size) + '-' + str(self.num_units_LSTM) + '-' + str(self.num_layers) + '-' + str(self.activation_decoder) + '-' + str(self.activation_inference) + '-' + str(self.input_dim) + '-' + str(self.num_units_inference)+'-'+ str(1) + '.csv'



