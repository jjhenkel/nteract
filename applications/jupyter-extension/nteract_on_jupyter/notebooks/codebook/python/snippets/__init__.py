import os
import re
import json
import glob

_SNIPPETS_PATH = os.path.dirname(
    os.path.realpath(__file__)
)

_SNIPPETS_GLOB = os.path.join(
    _SNIPPETS_PATH, '**/*.dl'
)

_SNIPPETS_FILES = {}
_SNR_IDX = 1

SNIPPETS = {}


class SnippetFunc:
    def __init__(self, snippet_template, childtype):
        self.template = snippet_template
        self.childtype = childtype
    
    def process_template(self, node, template, inputs):
        global _SNR_IDX

        nid = node.id if node.label is None else node.label

        replacements = {
            'rand(1)': 'r{}'.format(_SNR_IDX + 1),
            'rand(2)': 'r{}'.format(_SNR_IDX + 2),
            'rand(3)': 'r{}'.format(_SNR_IDX + 3),
            'rand(4)': 'r{}'.format(_SNR_IDX + 4),
            'rand(5)': 'r{}'.format(_SNR_IDX + 5),
            'rand(6)': 'r{}'.format(_SNR_IDX + 6),
            'rand(7)': 'r{}'.format(_SNR_IDX + 7),
            'rand(8)': 'r{}'.format(_SNR_IDX + 8),
            'fid()': 'fid',
            'fpath()': 'fpath',
            'type()': 'type_{}'.format(nid),
            'nid()': 'nid_{}'.format(nid),
            'gid()': 'gid_{}'.format(nid),
            'text()': 'source_text_{}'.format(nid),
            'sr()': 'start_line_{}'.format(nid),
            'sc()': 'start_col_{}'.format(nid),
            'er()': 'end_line_{}'.format(nid),
            'ec()': 'end_col_{}'.format(nid),
            'output("text")': 'out_text_{}'.format(nid),
            'output("name")': 'out_name_{}'.format(nid),
            'output("module_name")': 'out_module_name_{}'.format(nid),
            'output("child_type")': 'out_child_type_{}'.format(nid)
        }

        _SNR_IDX += 16
        outputs = {}

        if node.parent is not None:
            replacements['type(parent)'] = 'type_{}'.format(node.parent.id if node.parent.label is None else node.parent.label)
            replacements['nid(parent)'] = 'nid_{}'.format(node.parent.id if node.parent.label is None else node.parent.label)
            replacements['gid(parent)'] = 'gid_{}'.format(node.parent.id if node.parent.label is None else node.parent.label)

        if node.children is not None and len(node.children) == 1:
            child = node.children[0]
            replacements['type(child)'] = 'type_{}'.format(child.id if child.label is None else child.label)
            replacements['nid(child)'] = 'nid_{}'.format(child.id if child.label is None else child.label)
            replacements['gid(child)'] = 'gid_{}'.format(child.id if child.label is None else child.label)
        elif node.children is not None and len(node.children) > 1:
            for i, child in enumerate(node.children):
                replacements['type(child{})'.format(i+1)] = 'type_{}'.format(child.id if child.label is None else child.label)
                replacements['nid(child{})'.format(i+1)] = 'nid_{}'.format(child.id if child.label is None else child.label)
                replacements['gid(child{})'.format(i+1)] = 'gid_{}'.format(child.id if child.label is None else child.label)

        for match in re.findall(r'{{.*?}}', template):
            cleaned = match.replace('{', '').replace('}', '')
            template = template.replace(
                match, replacements[cleaned] if cleaned in replacements else '???'
            )
            if 'output' in cleaned:
                cleaned_2 = cleaned.replace('output(', '')[1:-2]
                outputs[cleaned_2] = replacements[cleaned]
        
        template = template.strip() + ',\n'
        if len(outputs.items()) > 0:
            return template, outputs
        return template
    
    def __call__(self, node, **inputs):
        target = node if self.childtype == False else node.children[0]
        if None in self.template:
            return self.process_template(
                node, self.template[None], inputs
            )
        elif isinstance(target.type, list):
            for typ in target.type:
                if typ in self.template:
                    return self.process_template(
                        node, self.template[typ], inputs
                    )
        elif target.type in self.template:
            return self.process_template(
                node, self.template[target.type], inputs
            )

        return '/* Unmatched {} */'.format(str(node))


for file in glob.glob(_SNIPPETS_GLOB, recursive=True):
    key = file.replace(_SNIPPETS_PATH, '')
    with open(file, 'rt') as fh:
        _SNIPPETS_FILES[key] = fh.read()
    
    SNIPPETS['/prelude.dl'] = lambda: _SNIPPETS_FILES['/prelude.dl']

for key, template in _SNIPPETS_FILES.items():
    rules_by_type = {}

    line_idx = 0
    choices = None
    childtype = False
    segment = ''
    in_comment = False
    paren_tempo = 0
    template_lines = template.split('\n')
    while line_idx < len(template_lines):
        # Fetch line
        og_line = template_lines[line_idx]
        line = og_line.strip()
        line_idx += 1

        # Check blank
        if len(line) <= 0:
            continue

        # Ignore comments
        if in_comment or (len(line) > 1 and line[:2] == '/*'):
            in_comment = (line[-2:] != '*/')
            continue

        # Check for '<<$type' directives
        if len(line) > 7 and line[:7] == '<<$type':
            choices = line.replace(
                '<<$type', ''
            ).replace('$>>', '').replace('(', '')
            paren_tempo += 1
            choices = json.loads(choices.strip())
            segment = '(\n'
            continue
        elif len(line) > 12 and line[:12] == '<<$childtype':
            choices = line.replace(
                '<<$childtype', ''
            ).replace('$>>', '').replace('(', '')
            paren_tempo += 1
            choices = json.loads(choices.strip())
            segment = '(\n'
            childtype = True
            continue

        if paren_tempo == 0 and line == '(':
            segment = '(\n'
            paren_tempo = 1
        elif paren_tempo > 0:
            segment += og_line + '\n'
            if '(' in line:
                paren_tempo += 1
            if ')' in line:
                paren_tempo -= 1

        if paren_tempo == 0 and len(segment.strip()) > 0:
            if choices is None:
                rules_by_type[None] = segment
            else:
                for c in choices: 
                    rules_by_type[c] = segment
            choices = None
            paren_tempo = 0
            segment = ''
        
    SNIPPETS[key] = SnippetFunc(rules_by_type, childtype)
