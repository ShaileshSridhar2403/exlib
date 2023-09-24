import torch.nn as nn


class TextVisualizer(nn.Module):

    def __init__(self, tokenizer, 
                 postprocess=(lambda ids, tokenizer: tokenizer.convert_ids_to_tokens(ids)),
                 normalize=False):
        """Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.postprocess = postprocess
        self.normalize = normalize
        self.style = f'''
        <style>
        .highlighted {{
            color: black;
        }}
        .sent {{
            margin-bottom: 5px;
        }}
        </style>
        '''
        # background-color: rgba(173, 216, 230, var(--alpha));

    def forward(self, X, Z, label=None):
        predicted_tokens_batch = [self.postprocess(ids, self.tokenizer) 
                                  for ids in X]
        if label is not None:
            label = label.cpu().numpy()
        htmls = self.highlight_sentences(predicted_tokens_batch, 
                                         Z.cpu().numpy(),
                                         label,
                                         self.normalize)
        return htmls
    
    def save(self, htmls, filepath):
        with open(filepath, 'wt') as output_file:
            output_file.write(self.style)
            for line in htmls:
                output_file.write(f'<div class="sent">{line}\n<div>')

    def highlight_sentences(self, sentences, weights_all, 
                            label=None,
                            normalize=False):
        results = []
        for i in range(len(sentences)):
            results.append(self.highlight_sentence(sentences[i], 
                                                   weights_all[i],
                                                   label[i],
                                                   normalize))
        return results

    def highlight_sentence(self, tokens, weights, 
                           label=None,
                           normalize=False):
        """
        Args:
            tokens (list of str): sentence that is already split into tokens
            weights (list of float between 0 and 1): weight for sentence
        """
        # words = sentence.split()
        
        # Normalize weights to range [0, 1]
        weights = weights.reshape(-1)
        if normalize:
            try:
                pad_start = tokens.index('[PAD]')
                tokens = tokens[:pad_start]
                weights = weights[:pad_start]
            except:
                pass
            
            max_weight = max(weights)
            min_weight = min(weights)
            if max_weight > min_weight:
                weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
        
        highlighted_words = []
        for token, weight in zip(tokens, weights):
            r = int(205 * (1 - weight)) + 50  # Red component increases as weight decreases
            b = int(205 * weight) + 50  # Blue component increases as weight increases
            highlighted_words.append(f'<span class="highlighted" style="background-color: rgb({r}, 50, {b});">{token}</span>')
        
        if label is not None:
            highlighted_words.append(f'<span style="padding-left: 10px">{int(label)}</span>')
            # highlighted_words.append(f'<span class="highlighted" style="--alpha: {weight:.2f};">{token}</span>')
        # Combine the words and add the shared styles
        return ' '.join(highlighted_words)
