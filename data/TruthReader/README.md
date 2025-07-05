---
license: apache-2.0
task_categories:
- question-answering
- text2text-generation
language:
- en
- zh
size_categories:
- 1K<n<10K
---


# Dataset Card for TruthReader
This dataset is used to train the response generator in TruthReader framework.

## Dataset Description

- **Homepage:** N/A
- **Repository:** [GitHub](https://github.com/HITsz-TMG/TruthReader-document-assistant)
- **Paper:** To be released
- **Leaderboard:** N/A
- **Point of Contact:** -

## Dataset information
| type | language | Source | Annotator | #sample |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Multi-document Synthesis | zh | WeiXin Articles | ChatGPT | 387 |
| Single-document Summary | zh,en | WeiXin Articles, Wikipedia | ChatGPT | 561 |
| QA Created | zh | Multi-domains | ChatGPT | 1,482 |
| WebCPM | zh | Web | Human | 897 |
| RefGPT | zh,en | Baidu Baike, Wikipedia | GPT-4 | 3,708 |

## Dataset columns
The examples have the following structure per configuration:

<details><summary> Example from train_zh </summary><hr>

```json
{
    "question": "恐龙种类及演化历程",
    "answer": "恐龙种类及演化历程是古生物学和演化生物学领域的重要研究内容[1]。根据提供的文章，我们可以了解到一些关于恐龙种类及演化历程的信息[1]。1. 文章提到的发现的新物种奇异福建龙（Fujianvenator prodigiosus）属于鸟翼类恐龙[1]。鸟翼类恐龙是鸟类起源和演化研究的重点，它们至少在晚侏罗世就和非鸟类兽脚类恐龙发生分化[1]。2. 已知的侏罗纪鸟翼类恐龙仅有近鸟龙和其相似物种，但物种数目稀少，地理分布单一[1]。而奇异福建龙的发现填补了侏罗纪鸟翼类恐龙在时间和空间上的空白，是侏罗纪最晚期和地理位置最南的鸟翼类恐龙[1]。3. 奇异福建龙的体型结构和其他鸟翼类相近，在体型空间上介于恐龙和鸟类之间[1]。它的前肢与始祖鸟相似，而后肢的构造则与其他已知的中生代恐龙和鸟类有明显区别[1]。奇异福建龙的发现显示了鸟类起源初期的生态多样性[1]。它被认为是一类善于奔跑的，或者生活在水边的小型兽脚类恐龙，与学界关于鸟类起源伊始生态习性演化的认识有所不同[1]。综上所述，奇异福建龙的发现为我们了解鸟类起源和演化提供了重要的线索，填补了侏罗纪鸟翼类恐龙的空白，展示了鸟类起源初期的生态多样性[1]。然而，关于恐龙种类及演化历程的研究还需要进一步深入，通过更多的发现和研究来完善我们对恐龙的认识[1]。",
    "history": [],
    "documents": [
        {
            "title": "【科研速报】|《自然》：中国发现新的鸟翼类恐龙和侏罗纪陆相动物群",
            "document": "鸟翼类恐龙\u0003侏罗纪最晚期  地理位置最南\u0003NATURE\u0003”\u0003奇异福建龙和政和动物群生态复原图（赵闯绘制）\u00039月6日，《自然》（Nature）发表了中国科学院古脊椎动物与古人类研究所（以下称“古脊椎所”）王敏团队和福建省地质调查调查研究院（以下称“福建地调院”）合作完成的有关福建省内中生代地层和古脊椎动物的研究成果，报道了世界上侏罗纪最晚期和地理位置最南的鸟翼类恐龙，以及大量其它脊椎动物，并结合年代地层和生物地层等工作，建立距今1.48–1.5亿年前的陆相生物群“政和动物群”。古脊椎所王敏为通讯作者和共同第一作者，福建地调院徐立明为第一作者。\u0003有关鸟类的起源和演化长期以来都是演化生物学讨论的重点，鸟类至少在晚侏罗世就和非鸟类兽脚类恐龙（以下称“兽脚类恐龙”）发生分化。学术界将“包括所有鸟类，但不包括恐爪龙类的最广义类群”定义为鸟翼类（Avialae），而鸟类（Aves）则指的是现代鸟类及其近亲。因此，侏罗纪的鸟翼类对研究鸟类的起源、关键形态和生物学特征的演化至关重要。已知的侏罗纪鸟翼类仅有近鸟龙和其相似物种，不仅物种数目稀少，而且地理分布单一（多在我国东北地区的燕辽生物群，距今1.66–1.59亿年），这与白垩纪早期出现的大量鸟类在时间上有长达三千万年的空白。\u0003院地合作，三年苦作，终结硕果\u00032021年，古脊椎所的尤海鲁研究员和福建地调院在福建省内进行古脊椎动物化石的调查工作。同年10月开始，王敏研究员带领古脊椎所的野外团队和福建地调院在多个晚中生代盆地开展大规模野外发掘，发现上百件包括鱼类、两栖类、龟鳖类等脊椎动物化石，但却未见恐龙和鸟类的踪影，失望和焦虑影响着每一个人。2022年10月23日，野外团队在政和晚侏罗世地层发现了一件保存近乎完整的恐龙化石，历经三年，累计发掘 200余天，终结硕果（图1）。2023年2月17日，为进一步巩固合作，古脊椎所与福建地调院签订战略合作框架协议（图2）。\u0003图1: 古脊椎所和福建地调院联合考察队发现奇异福建龙正型标本（前排左起：李岩、冯久桐、董丽萍、王敏、苗嵩、冯文清；中排左起、李宝贵、林虓、汤建荣、王林昌；后排：黄代和、黄代栋、陈官明）\u0003图2: 古脊椎所与福建地调院签订战略合作框架协议和多次野外考察\u0003“从0到1”的发现：福建第一龙\u0003经过长达一年的修复和分析研究，研究团队认为新物种属于鸟翼类，并将其命名为奇异福建龙（Fujianvenator prodigiosus），这也是福建省内首次发现恐龙化石。福建地调院徐立明带队开展的综合地质考察和同位素测年工作，将福建龙生活的时间限定为晚侏罗世提通期。古脊椎所王敏等通过古地理位置的复原，确定了福建龙是目前已知地理位置最南的侏罗纪鸟翼类（图3）。\u0003图3: 奇异福建龙正型标本，分支系统树和古地理图（王敏供图）\u0003奇异福建龙的发现弥补了鸟类起源在时间和空间上的部分空白，并显示高度镶嵌演化的形态特征：前肢与始祖鸟相似，腰带的耻骨和坐骨又分别具有伤齿龙类和近鸟龙的典型特征，而后肢更是如此，说明镶嵌演化深刻影响鸟类起源之初的特征演化。包括简约法和贝叶斯法在内的系统发育分析显示，奇异福建龙与近鸟龙类构成单系类群，是鸟翼类最早分异的一支。奇异福建龙最为特殊的是其后肢构造：股骨短，胫骨和蹠骨细长。结合比较分支系统学的分析，王敏等发现奇异福建龙的体型结构和其他鸟翼类相近，在体型空间上介于恐龙和鸟类之间，而这样的相似性更多的是演化相对保守的前肢造成的。因为如果仅比较后肢，奇异福建龙则明显区别于所有已知的中生代恐龙和鸟类（图4）。\u0003图4: 奇异福建龙的体型和奔跑能力与其他中生代恐龙的比较（王敏供图）\u0003在四足动物中，相对更长的远端肢骨能够增加步长，所以多见于善于奔跑的动物，也在部分涉禽中常见，研究人员认为奇异福建龙是一类善于奔跑的，或者生活在水边的小型兽脚类恐龙。这样的生活习性完全区别于学界关于鸟类起源伊始生态习性演化的认识，多数认为适应树栖是“主调”，而奇异福建龙的发现增加了原始鸟翼类的生态多样性。\u0003奇异福建龙三维模型（任文熠制作）\u0003“南政和，北燕辽”：\u0003相似的构造背景，不同的生物群表现\u0003除了奇异福建龙外，古脊椎所和福建地调院组成的考察团队还发现了大量保存完好的爬行动物，包括水生/半水生的龟鳖类、离龙类。基于如此高的化石丰度和多样性，以及确切的年代学框架，研究人员将其命名为“政和动物群”（Zhenghe Fauna），这也是目前全球已知侏罗纪最晚期，地理位置最南的保存有鸟翼类的动物群（图5）。从晚侏罗到早白垩，受古太平洋板块俯冲，我国西南地区岩石圈发生强烈伸展，形成广泛分布的断陷盆地和大规模火山活动，这样的构造背景和燕山运动A幕时期的华北地区是相似的，而后者与燕辽生物群的形成相关。虽然政和的野外考察刚起步，但是已经显示出与燕辽生物群的差异，前者保存了大量真骨鱼类、离龙类和龟鳖类，这些在燕辽生物群还鲜有报道。这些化石多保存在黑色碳质泥岩或页岩中，结合野外区域考察，研究人员推测政和动物群为类似沼泽相的环境，这也与燕辽生物群不同。\u0003图5: 政和动物群剖面及其岩性柱状图（王敏、徐立明、汤建荣供图）\u0003Nature论文作者包括古脊椎所的董丽萍、徐星、尤海鲁、张驰、周忠和，以及福建地调院的陈润生、林敏、汤建荣等。该研究得到了国家自然科学基金杰出青年基金、基础科学中心项目“克拉通破坏与陆地生物演化”、中国科学院前沿科学重点研究计划从“0到1”原始创新十年择优项目、腾讯探索奖、福建省自然资源厅“闽西地区晚中生盆地地质遗迹及古生物化石资源调查”和福建省地质矿产勘查开发局“福建恐龙化石赋存沉积环境研究”等项目的资助。\u0003奇异福建龙发掘之旅（冯久桐制作）\u0003IVPP\u0003主要作者简介\u0003王敏\u0003中国科学院古脊椎动物与古人类研究所研究员。\u0003主要研究鸟类的起源和早期演化，尤其是中生代这一时期鸟类是如何从恐龙演化而来，并最终演化出现代鸟类的主要特征。\u0003本文作者：王敏团队\u0003文稿校审：侯韡鸿\u0003排版编辑：肖潇\u0003本文来自中国科学院古脊椎动物与古人类研究所，\u0003未经授权不得转载。\u0003如有需要请联系dxc@ivpp.ac.cn\u0003"
        }
    ],
    "type": "Multi-document Synthesis"
}
```

</details>

* `question`: the question that LLM or human generates, used to retrieve documents and seek answers.
* `answer`: the answer to the question given retrieved documents. If the question is unanswerable, a refusal answer is provided.
* `history`: the context of the current question-answering session.
* `documents`: documents retrieved from the internet or knowledge bases.
  * `title`: the title of the retrieved documents.
  * `document`: the filtered version of the retrieved documents (after pre-processing).
* `type`: sample type, one of `Multi-document Synthesis`, `Single-document Summary`, `QA created`, `RefGPT` and `WebCPM`.

## Considerations for Using the Data
The dataset is unbalanced since we pay more attention on Chinese language. A further balancing and filtering approach may be useful.


# References
```
@misc{truthreader,
  author = {Xinshuo Hu and Zetian Sun and Dongfang Li and Shaolin Ye and Zifei Shan and Qian Chen and Baotian Hu and Min Zhang},
  title = {TruthReader: Towards Trustworthy Document Assistant Chatbot with Reliable Attribution},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HITsz-TMG/TruthReader-document-assistant}},
}
```