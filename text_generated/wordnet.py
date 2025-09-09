import nltk
from nltk.corpus import wordnet2022 as wn


# nltk.download("wordnet2022")
# nltk.download("omw-1.4")

def normalize_key(class_name: str) -> str:
    """把类别名转成 WordNet 喜欢的下划线风格，比如 'pickup truck' -> 'pickup_truck'"""
    return class_name.strip().replace(" ", "_")

def pick_synset(class_name: str, prefer_pos=wn.NOUN, manual_override=None): #class_name:传入的类别名， prefer_pos:首选词性（默认为名词）
    """
    选一个最合理的 synset：
    - 手工覆盖 > 精确同名 > 下划线风格 > 退而求其次（第一个名词义项）
    可以在 manual_override 里配置易歧义词，比如 {'seal': 'seal.n.02'}。
    """
    if manual_override and class_name in manual_override:
        s = wn.synset(manual_override[class_name])
        return s

    cands = wn.synsets(class_name, pos=prefer_pos)
    if not cands: #如果cands为空
        cands = wn.synsets(normalize_key(class_name), pos=prefer_pos)
    if not cands:
        return None

    # 简单启发式：优先 noun.*，定义更长、例句更多、meronym/holonym更丰富者
    def score(ss):
        score = 0
        score += 3 if ss.pos() == 'n' else 0
        score += len(ss.definition())
        score += len(ss.examples()) * 5
        score += len(ss.part_meronyms()) * 4 + len(ss.substance_meronyms()) * 2
        return score

    cands.sort(key=score, reverse=True)
    return cands[0]

def immediate_hypernym(ss):
    """最近上位（可能为空）"""
    hs = ss.hypernyms() 
    return hs[0] if hs else None

def coordinate_terms(ss, limit=6):
    """
    同级易混：= 最近上位的所有下位 - 自己
    返回若干个代表性兄弟类（用可读的 lemma）
    """
    h = immediate_hypernym(ss)
    if not h: #如果上位词为空
        return []
    bros = []
    for hy in h.hyponyms(): #获取上位词的所有下位词
        if hy == ss:
            continue # 跳过自己
        # 每个兄弟取更易读的名字（第一个 lemma）
        bros.append(hy.lemmas()[0].name().replace("_", " "))
    # 去重+裁剪
    seen, uniq = set(), []
    for b in bros:
        if b not in seen:
            seen.add(b); uniq.append(b)
    return uniq[:limit]

def meronyms_readable(ss, limit=8):
    """
    部件信息（part/substance meronyms），返回可读的名词列表
    WordNet 有时比较稀疏，判空即可
    """
    parts = [m.lemmas()[0].name().replace("_", " ") for m in ss.part_meronyms()] #组件
    subs  = [m.lemmas()[0].name().replace("_", " ") for m in ss.substance_meronyms()] #材质
    # 去重并裁剪
    allm = []
    for x in parts + subs:
        if x not in allm:
            allm.append(x)
    return allm[:limit]

def synset_lemmas_readable(ss):  #获取当前词义的同义词，并转化为可读形式
    return [l.name().replace("_", " ") for l in ss.lemmas()]

def build_prompt_from_wordnet(class_name: str, manual_override=None, use_wordnet=True):
    if use_wordnet:
        ss = pick_synset(class_name, manual_override=manual_override)
        if not ss: #如果不为空
            prompt = f"""你是一名视觉判别专家。目标类别：{class_name}
    未能在 WordNet 中找到对应名词义项；请基于常识给出“判别要点清单（3–6条）+一段式描述（120–200字）”，
    尽量聚焦形态与部件，避免使用环境/用途/地域作为判据。"""
            return prompt, {"class": class_name, "found": False}

        synset_id = ss.name()    # e.g. 'seal.n.02'
        gloss = ss.definition() #定义
        lexname = ss.lexname() #领域
        lemmas = synset_lemmas_readable(ss) #这个词义的同义词有哪些？如果是多个词转化为下划线形式

        hyper = immediate_hypernym(ss) #获取这个词义的最近上位词
        hyper_readable = hyper.lemmas()[0].name().replace("_", " ") if hyper else "（无）" #如果有上位词则变为可读形式，反之为空

        coords = coordinate_terms(ss, limit=6) #同级易混词：取出最近上位词的所有下位词，去掉自己，然后列出前limit个
        meros  = meronyms_readable(ss, limit=8)#获取这个词义的部件并转化为可读形式

        # 模板：判别性 + 对比驱动
        prompt = f"""
        请根据以下 WordNet 语义信息生成类别判别性描述：

        【目标类别】
        - 名称与同义词：{", ".join(lemmas) if lemmas else class_name}
        - WordNet 词义：{synset_id}（领域：{lexname}）
        - 定义：{gloss}

        【层级信息】
        - 最近上位类别：{hyper_readable}
        - 同级易混类别：{", ".join(coords) if coords else "（无，若无则请基于上位类推测常见混淆）"}

        【结构部件】
        - 主要部件：{", ".join(meros) if meros else "（WordNet 未给出明显部件，请从常见外形部件角度补全）"}

        【输出要求】
        1) 一段式描述：根据正向线索（能识别该类的稳定形态/结构/比例）和排除线索（与同级易混类别的结构差异），生成该类别的高区分度描述（50 字左右）。
        """.strip()

        meta = {
            "class": class_name,
            "found": True,
            "synset_id": synset_id,
            "gloss": gloss,
            "lexname": lexname,
            "lemmas": lemmas,
            "hypernym": hyper_readable,
            "coordinate_terms": coords,
            "meronyms": meros,
        }
        return prompt, meta
    else:
        # 如果不使用 WordNet，直接返回一个通用的提示模板
        prompt = f"""
        Task: Generate a visually discriminative description for the image category "{class_name}".

        Requirements:
        1. Describe only morphological, structural, proportional, and texture-related attributes that can be directly observed from an image of the target class.
        2. Emphasize features that differentiate the target class from other visually similar categories.
        3. Avoid non-visual information such as habitat, usage, behavior, seasonality, or geographic distribution.
        4. Mention color only if it is common but not guaranteed, and link it to structural context (e.g., “dark stripes on light fur” rather than “brown”).
        5. Limit the output to a single concise paragraph (30 words), maintaining precise and formal technical language.

        Output: A paragraph describing the key visual characteristics of the category "{class_name}" in accordance with the above requirements.
        """.strip()
        meta=None

        return prompt,meta
##############################################################################
# 2) 批量生成：输入你的 {label: class_name} 字典，输出 {label: prompt}
##############################################################################

def build_prompts_for_dict(label2name: dict, manual_override=None,use_wordnet=True):
    prompts = {} 
    metas = {} 
    for label, cname in label2name.items():
        p, m = build_prompt_from_wordnet(cname, manual_override=manual_override,use_wordnet=use_wordnet)
        prompts[label] = p
        metas[label] = m
    return prompts, metas

##############################################################################
# 3) 用法示例（替换成你的真实字典）
##############################################################################

if __name__ == "__main__":
    # 你的标签->类别名字典
    label2name = {
        0: "cowboy boot" #歧义：海豹/印章
    }
    use_manual_override = False

    # 建议：为高歧义类别加“手工覆盖”
    if use_manual_override == True:
         manual_override = {
        "seal": "seal.n.09", # 海豹（动物）
                            }
    else:
        manual_override = None

    prompts, metas = build_prompts_for_dict(label2name, manual_override=manual_override)

    # 打印看看
    for k in sorted(prompts):
        print("="*80)
        print(f"Label {k}  |  Class: {label2name[k]}")
        print(prompts[k])