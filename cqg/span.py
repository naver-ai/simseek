import re
import nltk


class SpanMapper:
    def __init__(self, text):
        text = re.sub('\s+', ' ', text)
        self.text = text
        self.sents = nltk.sent_tokenize(text)
        maps = []
        start = 0
        for sent in self.sents:
            end = start + len(sent)
            maps.append([start, end])
            start = end + 1
        self.maps = maps
        
    def align(self, span_start, span_text):
        span_end = span_start + len(span_text)
        find = False
        for i, m in enumerate(self.maps):
            ms, me = m
            
            if span_start >= ms and span_end <= me:
                find = True
                span_text = self.sents[i]
                span_start = ms
                break
        return span_start, span_text


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


class AnswerSpanAligner:
    
    def __init__(self, passage, tokenizer, get_offset=False):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        for c in passage:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False

            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset
        self.tokenize(tokenizer)
        if get_offset:
            self.offset = tokenizer(passage, return_offsets_mapping=True)['offset_mapping']
        
    def _get_span_in_words(self, start_position_character, answer_text):
        start_position = self.char_to_word_offset[start_position_character]
        end_position = self.char_to_word_offset[
            min(start_position_character + len(answer_text) - 1, len(self.char_to_word_offset) - 1)
        ]
        
        return start_position, end_position
    
    def get_span_in_subwords(self, start_position_character, answer_text, tokenizer):
        start, end = self._get_span_in_words(start_position_character, answer_text)
        start = self.orig_to_tok_index[start]
        if end < len(self.orig_to_tok_index) - 1:
            end = self.orig_to_tok_index[end + 1]
        else:
            end = len(self.all_doc_tokens)
            
        start, end = _improve_answer_span(
            self.all_doc_tokens, start, end, tokenizer, answer_text
        )
        return start, end
    
    def tokenize(self, tokenizer):
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        offset = []
        for (i, token) in enumerate(self.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        self.tok_to_orig_index = tok_to_orig_index
        self.orig_to_tok_index = orig_to_tok_index
        self.all_doc_tokens = all_doc_tokens
    
    def get_highlight_by_answer(self, tokenizer,
                                start_position_character, answer_text, max_passage_length, hl_token="<hl>"):
        start, end = self.get_span_in_subwords(start_position_character, answer_text, tokenizer)
        
        highlight = [hl_token] + self.all_doc_tokens[start:end+1] + [hl_token]
        new_doc = self.all_doc_tokens[:start] + highlight + self.all_doc_tokens[end+1:]
        
        if len(new_doc) > max_passage_length:
            gap = len(new_doc) - max_passage_length
            
            if ' '.join(highlight) in ' '.join(new_doc[:-gap]):
                new_doc = new_doc[:-gap]
            else:
                new_doc = new_doc[gap:]

        return new_doc
