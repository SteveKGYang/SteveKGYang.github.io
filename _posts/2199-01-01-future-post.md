---
title: 'A brief survey about image caption (2010-2018)'
date: 2018-12-01
permalink: https://blog.csdn.net/qq_41533506/article/details/84671195
tags:
  - image caption
  - survey
---

第一次写image caption简要综述类文章，对image caption任务进行了一个简单调研。如发现某些地方有问题希望大家批评指正，共同进步。

为一张图片产生一个描述被称为image caption任务。在编码端Image caption任务要求我们识别图中重要目标，其属性和相互关系。而在解码端要求我们产生语义和语法和正确的句子。本文对image caption近年发展历程和最新进展做了简单调研，并做一个简单的总结。由于当前深度学习方法在image caption中占主导地位，故本文主要关注基于深度学习的方法。

一. 图像处理方法简历
早期的图像处理方法基于传统机器学习，包括使用一些图像处理的算子来提取出图像的特征，使用支持向量机等进行分类以得到图像中的目标，再将得到的目标及其属性作为生成句子的依据。这样的做法在实际应用中并不理想。具有代表性的做法有Kulkarni et al.[3], Farhadi et al.[4]等较早期的工作。

深度学习促进了计算机视觉的迅速发展。图像编码和特征提取大大受益于CNN的发展。随着VGG等深度CNN编码器的出现，图像识别等任务准确率迅速提高。由于CNN强大的图像特征提取能力，在image caption任务中使用深度CNN网络作为图像特征编码器成为主流的做法。谷歌在2014年提出Neural Image Caption模型算是这种方法的开山之作。随后的Neural Talk等对image caption发展有较大影响的模型几乎都沿用了这个基本框架。

在video caption领域，传统编码端方法大同小异，主要的区别是video caption提取的特征随时间发生变化。除此之外，还有一张被称为3D特征提取[39]的方法，其思想是将视频的每一帧图像在channel维度合并起来，在进行3D的卷积。其主要目的是获得视频不同帧之间的相互关系。

视频还有一个特点就是其包含音频特征。谷歌提供了一个音频特征数据集，可用于这方面的研究。

二. 文字生成方法简历

在上文中提到，解码端主要任务是获得具有正确语法和正确语言的句子。针对这些目标，Image caption任务主要有3种生成文字的方法：基于模板填充的方法，基于检索的方法和基于生成的方法。

基于模板填充的方法主要指的是在人为规定的一系列句法模板中留出部分空白，然后再基于提取出的图像特征获得目标，动作及属性，将它们填充进入空白，从而获得对某一图像的描述。这种方法的代表有Li et al.[1], Kulkarni et al.[2]等。这种方法保证了语义和句法正确性。然而，完全确定的模板无法产生多样性的输入，故现在这种方法使用较少。

基于检索的方法指的是将大量的图片描述存于一个集合，再通过比较带描述图片和训练集中图片描述的相似性获得一个待选句集，再从中选取该图片的描述。这种方法保证了句法正确性，然而无法保证语义正确性，也无法对新图片进行准确的描述。

目前使用较多的是基于生成的方法。基于生成的方法大致流程是先将图像信息编码后作为输入送入语言模型，再利用语言模型产生全新的描述。绝大部分基于深度学习的image caption方法使用基于生成的方法，也是目前效果最好的image caption模型上普遍应用方法。它在句法正确性，语义准确性和对新图片的泛化能力上都达到了较好的效果。

在video caption领域，早期的解码方法基于句法结构，先预测主干关键词，如句子中的实体，动词等，再补全句子。Venugopalan et al.[33] 首次提出基于CNN 和RNN的seq2Seq生成方法，使用LSTM作为解码器获得caption。

三. Image caption代表性工作综述

下面主要按照重要工作和思想的归类对image caption领域具有代表性的工作进行简单总结,总结的重点是基于深度学习的方法。

1. Encoder-Decoder框架

谷歌在2014年提出了Neural Image Caption Generator[5]。不同于以往的基于规则和分类获取特征的方法，其受大获成功的机器翻译的模型影响，将原机器翻译模型中的用于提取原语言特征的RNN改为基于CNN的InceptionNet用于提取图像特征，而使用RNN作为解码器接受CNN提取出的图像，其中RNN也可替换为LSTM 或GRU等，以获得更好的长期记忆。而几乎与其同时提出的还有斯坦福大学的Neural Talk[6], 其基本架构与谷歌的模型几乎一模一样，唯一的区别是其使用的图像特征提取器是VGGnet。

以上两项工作首次提出image caption的编码-解码基本框架，可以被称作相关工作的开山之作。这种Encoder-Decoder模型对图像理解方向的工作影响巨大，向image caption任务中引入这样的架构已成为主流之一。其后有关工作主要是在其基础上作变化和提高。
先介绍一些编码端的主要改进。

微软在2015年提出了一种编码端改进方法，发表在论文From Captions to Visual Concepts and Back[10]中。该方法使用多实例训练训练一个词探测器，用于为每一张图片产生一系列可能出现在caption中的词语。再将获得的词语作为输入使用语言模型产生一系列关于该图片的描述句子，最后从中选择结果句子。这种通过提取关键词作为输入产生句子的方法无疑为接下来介绍的结合图像和语义的编码方法提供了借鉴。

Li et al.[9] 在2018年提出了一种新的特征提取方法。在提取图像特征时，通过目标检测算法获得一系列的目标检测框作为图像特征，并同时以图像特征为输入训练一个属性检测器。属性作为高层语义特征，和提取的图像特征一起作为经过特殊设计的Visual-Semantic LSTM的输入，再进行解码。这种使用目标检测使得输入特征更加“稠密”，而不是像以前直接输入整张图片，从而获得一种类似视觉注意力的效果。2018年发表于CVPR的工作Bottom-Up and Top-Down Attention[8]也使用了类似的编码器结构。

相比于编码端工作的创新，在解码端的改进更令人印象深刻。

Wang et al.在2017年提出了一种新型的decoder结构。这种被称为Skeleton-Attribute decoder的解码器由一个Skel-LSTM和一个Attr-LSTM构成，其中前者使用CNN提取出的图像特征获取一个主干句子，再使用后者为每一个主干句子中的词语获得一系列的属性词，再将两部分词语合成最终的caption。类似的工作还有Neural baby talk。受基于句子模板填充的baby talk[11]的启发，Lu et al.[12]于2018年提出了一种基于模板生成和填槽的image caption方法。其主要思想是将产生句子中的词语分为实体词与非实体词两个词表，句子模板由一个语言模型获得，其词语来自非实体词表。实体词由目标检测方法直接由图像中获得，再用于填充句子模板中的空槽，形成一个句子。这种方法开创性的使用神经网络来提取句子模板，从而成功解决了在第二部分我们对传统基于模板填充的方法缺乏多样性输入的问题。

类似于以上工作的将解码器分离的思路，Mathews et al.[15] 为了获得风格化的image caption结果，使用两个解码器，第一个称为term generator，使用CNN图像特征作为输入，通过GRU获得一系列基本语义对，由词语-属性组成。其后再将term generator获取的基本语义输入language generator，产生最后的输出。其中language generator用双向GRU编码按顺序排列的基本语义，再使用新的GRU进行解码。

进一步发展以上文章对于解码器的改进，Gu et al.[17]进一步提出了逐步求精的stack caption的思想。其主要创新点在于使用一个粗粒度的解码器和多个细粒度的解码器，其中粗粒度解码器接受图像特征作为输入，并获得粗粒度的描述结果。接下来在每一个阶段都有一个细粒度的解码器进行更精细的解码，其输入来自于上一阶段解码器的输出结果和图像特征，并使用attention机制，从而使得细粒度解码器在每一阶段对粗粒度产生结果的不同方面进行扩展，最终获得较详细的结果。

以上的工作充分说明在解码端使用层级的或切分的解码思路可以显著提高image caption的效果，这样的解码思路也较为符合人类的思维模式，可解释性较强。可以预见这样的层级结构在未来有成为主流的潜力。

此外，还有一种完全不同于传统基于RNN进行解码的工作，即使用卷积神经网络作为解码器进行caption。其代表是2018年发表在CVPR上的工作Convolutional image caption[16]。这项工作激进地使用Masked CNN代替传统RNN作为解码器进行解码。在编码阶段，每一步都将词语和提取的图像特征输入卷积编码器，再使用卷积解码，最后再使用softmax获得词概率。这种使用CNN的方法避免了RNN的时序限制，在相同参数量下获得了更快的训练速度。

让产生的caption获得超出Ground truth描述物体的限制而自动探测并描述图片中未被提及的目标一直是一个研究热点。比如说Yao et al.[18] 在2017年将复制机制（copy mechanism）引入了image caption的解码端。其基本思想是在一张图片中同时引入传统的encoder-decoder模型的同时再在图像特征上训练一个目标检测器，再计算解码器LSTM的输出层ht和检测到的实体的相似性以决定是否将在该步对该实体进行复制。由于caption模型和目标检测模型词表有所不同，故引入copy机制可以在结果中引入原caption中没有的实体，以丰富image caption模型的语义。

接上段主题，在描述一张图片时，可能只利用图片上的知识是不够的。知识图谱的建立与应用是迅速发展的一个领域，因此在image caption中通过知识图谱引入外部知识是值得研究的一个方向。在这方面的尝试包括发表在EMNLP2018上的工作Entity-aware Image Caption[18]。该工作采用类似于Neural Baby Talk的方法，先利用encoder-decoder获得一个含有实体空槽的语言模板，再使用实体填槽。该工作采用将训练数据的标签相近的图片的描述作为上下文，从中抽取命名实体输入知识图谱，选择图谱中概率最高的实体组合作为插槽输入。这种引入外部知识的方法大大提高了语义丰富度。

随着image caption技术的发展，对单张图片特征的利用似乎已经不足以继续提高caption的效果了，解码端的多模型联合训练成为一个研究方向。代表性工作有Chen et al.[23] 在2018年开创性地提出的一种image caption模型：Groupcap模型。这项工作的启发来自于希望在caption过程中编码多张图像从而同时获得相似性和多样性。模型第一部分为一个视觉语法分析树，用于将图像特征建模到树的节点中。该过程的监督信号来自对于ground truth语句的parsing tree。模型的第二部分为结构化相关性和多样性限制模块，对于输入的图片三元组，他们之间的相似性和多样性由其parsing tree的叶节点实体决定。每张训练图片三元组除目标图片之外，另两张图片都有一个标签为positive或negative，表示其与目标图像是否相近。训练的目的是最大化同组图片相似性，最小化非同组图片相似性，对于多样性则目标相反。第三部分则是caption generation环节。三个部分联合训练，以期利用所有提取的特征获得最佳的输出。该多模型联合训练方法引入新图形模型获得分辨形似图片和不同图片的能力，增强了模型对图像特征的理解。

以上是在编码器-解码器基础上发展出的的一些代表性image caption工作。在编码端主要体现在目标检测的引入和关键词提取的引入上。在解码端体现在包括解码过程层次化，卷积网络解码和外部知识引入等创新方法上。可以预见在相当长的时间内图像理解的工作依然会以这样的基本结构进行延伸。

在video caption 领域，Pasunuru et al.[32] 将一个video caption任务与一个无监督的video重建任务，一个限定生成任务进行联合训练，共用编解码端，以求获得更好的效果。

2. 注意力机制的引入

注意力机制在机器翻译领域的成功引起了image caption领域对其的兴趣。 Xu et al.[7] 于2016年提出Show, Attend and Tell中提出将注意力机制应用于表征image caption的图像特征中。其基本思想是利用卷积层获取图像特征后，对图像特征进行注意力加权，之后再送入RNN中进行解码。该文章提出了两种attention机制：软注意力机制（soft-attention）和硬注意力机制（hard-attention）。软注意力机制对每一个图像区域学习一个大小介于0与1之间的注意力权重，其和为1，再将各图像区域进行加权求和。硬注意力机制则将最大权重置为1，而将其他区域权重置0，以达到仅注意一个区域的目的。在实际的应用中软注意力机制得到了更广泛的应用。由于其良好的效果和可解释性，attention机制已经成为一种主流的模型构件。

一种具有代表性的attention改进机制来自发表在CVPR 2017的工作Knowing when to look[21]。这项工作考虑传统的空间注意力机制不具有很好的决定什么时候从图像特征中获取新特征的灵活性，从而提出了“视觉岗哨”的概念。“岗哨向量”表示解码器记忆中已经获得的知识，而“岗哨门控”为一个门控机制，用于控制“岗哨向量“和attend之后的图像特征分别所占权重，将其加权相加后作为该时间步的解码向量。这样的方法使得模型获得决定每一步更多关注语言模型中获得的语义还是更多关注新的图片特征。实验结果证明这种方法拥有比传统attention机制更好的效果和可解释性。

另一种具有创新性的改进图像特征attention的机制来自于Anderson et al.[8] 于2018年提出的Bottom-Up and Top-Down Attention。其主要创新在于使用Faster R-CNN进行目标检测，获得对应检测目标和标签，达到Bottom-Up attention的效果。此外，其还在解码端使用了注意力LSTM层，对输入的图片特征根据输出的语言进行实时的注意力调整。这种attention模式使得模型能够更加关注图片中更明显和重要的目标的同时使得描述更有主次感，即对于图像中明显和重要的目标进行更多关注。

除图像特征的注意力机制之外,语言特征的注意力机制也是自然而然的研究方向。语言注意力机制关注一系列的语义概念，这些概念通常抽取自训练集中频率较高的的词语。通过在获取语句过程中对这些概念进行注意以达到丰富语义的效果。代表工作是2016年的Wu et al.[20]。

近期attention研究的一个热点是如何将语义和视觉attention结合起来使用。Liu et al.[21] 在2018年EMNLP会议上发表了被称为Sim Net的image caption模型，其创新之处在于使用了一种Merging Network来结合visual attention和semantic attention，其中使用名词作为候选主题，使用多实例学习方法来从图像特征中提取主题词汇作为semantic attention的对象。将输入图像特征进行input attention后同该时间输入的文字向量一起编码成为输入向量，再将主题词attend后的结果和输入编码一起经过MLP后送入M-Gate，跟它们一起输入的还有经过output attention的图像特征。M-Gate根据output attention对主题向量和输入向量再次进行加权，以决定该时间步对语义和图像哪方面有更多关注。这种对语义和图像特征的双重注意，使得产生句子的过程更加关注主题，获得更围绕主题的caption。

Attention机制由计算机视觉引入，在自然语言处理领域获得长足发展。而在image caption这样结合CV和NLP的领域，attention机制无疑是最有发展潜力的研究方向之一。

3. 生成对抗方法

传统的encoder-decoder结构在训练上多采用交叉熵作为损失函数，这样做的结果使模型趋向于在生成中产生更“宽泛”的描述，即使得不同图片的描述趋向泛化。显然这与image caption中使caption趋向多样化的目标相违背。基于这样的考虑，研究者开始在将在图像生成上有惊人表现的生成对抗网络引入image caption中。

生成对抗网络，简称为GAN，其基本思想类似于非零和博弈。其基本架构包括一个生成器和一个鉴别器。生成器的目标是最大化拟合真实数据的概率分布，使得产生的虚假样本“以假乱真”，而鉴别器的训练目标则是对真实数据和生成器产生的虚假数据进行分类，以期在训练中增强分辨虚假数据的能力。

在image caption中引入生成对抗结构的历史并不长，毕竟GAN本身历史也并不悠久。Dai et al.[24]在2017年的CVPR大会上提出了使用Conditional GAN 进行image caption的方法，其动力在于产生更富多样性的image caption。其核心结构类似于传统的GAN结构，由一个生成器和一个判别器组成。生成器使用传统的encoder-decoder结构，输入图像得到伪造的caption。值得提及的是，该文通过随机初始化生成器LSTM隐藏层向量z，通过控制该向量方差来控制为同一张图片产生的不同结果的多样性。而判别器通过LSTM在每一步随机接收真实caption和伪造的caption，并接受一个图像特征，用于为caption打分，以期正确的区分真实答案与伪造答案。这项工作引入了非常典型的GAN结构，并在实验中证明了相比于传统方法其的确有增强结果多样性的效果。

在2017年的ICCV会议上，另一项被称为Speaking the Same Language[25]的工作也引入了生成对抗训练结构。其主体结构与上一项工作差别不大。其主要创新之一在于利用了大部分数据集对于同一张图片有多个caption的特点，对一种新的判别器验证结构的引入。除传统的图像-文字距离判定以验证图像-语义相似度之外，还引入了对于同一张图片的不同caption之间距离的测量，从而在语义多样性上对结果进行判定，不同语句之间多样性越大则越接近真实结果。这种在判别过程中对语义多样性的量化，进一步提高了生成器在产生caption过程中对语义多样性的关注。

此外，一些使用对抗样本对image caption进行攻击以检测鲁棒性的工作，如Chen et al.[26] 使用图像对抗样本进行攻击，Shekher et al.[27] 通过使用语义对抗样本评价模型鲁棒性等工作，在评价模型方面提供了新思路。而Dai et al.[28]则使用对抗样本来训练模型，以期获得更多样和可靠的结果。

4. 强化学习方法

强化学习的方法在人工智能的各个领域有广阔的前景，将其应用于image caption也会自然而然解决一些棘手的问题。前文提到使用最大似然来训练存在一些问题如鼓励泛化的答案，损失函数和评估方法不一致等问题。使用强化学习方法直接最大化奖励则可以避免这些问题。

强化学习方法在seq2seq序列生成上的应用可以追溯至2016年Ranzato et al.[29]的工作，通过直接优化BLEU等评价指标来训练模型。此后，针对在image caption训练过程中对数似然与评价指标相关性不强的问题，Liu et al.[30] 提出直接将评价指标等如SPICE 和 CIDEr 作为奖励通过策略梯度来优化参数。针对序列生成中无法评价的问题，该文章使用蒙特卡洛方法对每一步进行整句sample并根据结果对每一步给与奖励。实验证明该方法在相同情况下比传统方法有更好的效果。这样的方法被广泛应用与之后的工作中，比如说前文中提到的Dai et al.[24]就在生成器优化过程中使用了几乎一样的优化策略，只是奖励变成了判别器的返回结果。

另一篇发表在CVPR2017上的基于深度强化学习的image caption方法[31] 将完整的强化学习方法引入生成过程。该工作将image caption看作决策生成的过程，输入图片和当前产生文字作为环境，将词表作为action space。策略网络是典型的encoder-decoder结构，而value网络结构类似于策略网络，由CNN编码图像，RNN编码产生的文字来输出奖励。奖励由视觉-语义编码决定。其中奖励的一部分来自句子编码，即RNN的最后一个隐层。视觉编码即CNN图像特征。通过联合训练视觉语义编码，最终的奖励由其欧氏距离决定。通过实验证明该工作取得了SOTA的结果。

在video caption任务上，Pasunuru et al.[40] 使用了强化学习方法在video caption中解决上述问题，即对数似然方法的局限性。该工作类似地使用评价标准直接作为奖励优化模型。除此之外该模型还提出了对CIDEr评价标准进行提升的CIDEnt奖励机制，提升了模型效果。

强化学习的特点决定其对于文本生成任务的训练是非常合适的，现有工作表明强化学习方法在提高生成质量和多样性，合理化训练方法等方面比传统方法更有优势。

5. 总结

本文首先总结了在image caption任务中主要的图像特征提取和文本生成方法，然后就 编码器-解码器结构，attention机制的发展，生成对抗方法的引入和强化学习方法的引入四个方面对image caption近年的主要发展做了简要总结。

参考文献：

[1] Siming Li, Girish Kulkarni, Tamara L Berg, Alexander C Berg, and Yejin Choi. 2011. Composing simple image descriptions using web-scalen-grams. In Proceedings of the Fifteenth Conference on Computational Natural Language Learning. Association for Computational Linguistics, 220–228.

[2].Girish Kulkarni,Visruth Premraj,Sagnik Dhar,Siming Li,Yejin Choi, Alexander C Berg, and Tamara L Berg.2011. Baby talk: Understanding and generating image descriptions. In Proceedings of the 24th CVPR. Citeseer.

[3] Kulkarni, Girish, et al. “BabyTalk: Understanding and Generating Simple Image Descriptions.” IEEE Conference on Computer Vision and Pattern Recognition IEEE Computer Society, 2011:1601-1608.

[4] Farhadi, Ali, et al. Every Picture Tells a Story: Generating Sentences from Images. Computer Vision – ECCV 2010. Springer Berlin Heidelberg, 2010:15-29.

[5] Vinyals, Oriol, et al. “Show and Tell: Lessons Learned from the 2015 MSCOCO Image Captioning Challenge.” IEEE Transactions on Pattern Analysis & Machine Intelligence 39.4(2016):652-663

[6] Karpathy, Andrej, and F. F. Li. “Deep visual-semantic alignments for generating image descriptions.” Computer Vision and Pattern Recognition IEEE, 2015:3128-3137.

[7] Xu, Kelvin, et al. “Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.” Computer Science (2015):2048-2057.

[8] Anderson, Peter, et al. “Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering.” (2017).

[9] Nannan Li, Zhenzhong Chen. Image Captioning with Visual-Semantic LSTM. (2018). IJCAI.

[10] Hao Fang ,Saurabh Gupta et al. From Captions to Visual Concepts and Back. (2015) CVPR

[11] Kulkarni, Girish, et al. “BabyTalk: Understanding and Generating Simple Image Descriptions.” IEEE Conference on Computer Vision and Pattern Recognition IEEE Computer Society, 2011:1601-1608.

[12] Jiasen Lu,Jianwei Yang et al. Neural Baby Talk. (2018) CVPR

[13] Ryan Kiros, Ruslan Salakhutdinov, and Rich Zemel. 2014. Multimodal neural language models. In Proceedings of the 31st International Conference on Machine Learning (ICML-14).595–603.

[14] Skeleton Key: Image Captioning by Skeleton-Attribute Decomposition. Yufei Wang, Zhe Lin, Xiaohui Shen, Scott Cohen, Garrison W. Cottrell.(2017).CVPR

[15] SemStyle: Learning to Generate Stylised Image Captions Using Unaligned Text. Alexander Mathews; Lexing Xie; Xuming He.(2018).CVPR

[16] Convolutional Image Captioning. Jyoti Aneja, Aditya Deshpande, Alexander Schwing (2018).CVPR.

[17] Stack-Captioning: Coarse-to-Fine Learning for Image Captioning, Jiuxiang Gu, Jianfei Cai, Gang Wang, Tsuhan Chen. (2018).AAAI

[18] Incorporating Copying Mechanism in Image Captioning for Learning Novel Objects. Ting Yao, Yingwei Pan, Yehao Li, Tao Mei (2017) CVPR

[19] Entity-aware Image Caption Generation. Di Lu, Spencer Whitehead, Lifu Huang, Heng Ji and Shih-Fu Chang. (2018).EMNLP.

[20] Qi Wu, Chunhua Shen,Lingqiao Liu,Anthony R.Dick, and Antonvanden Hengel. 2016. What value does explicit high level concepts have in vision to language problems? CVPR2016, Las Vegas, NV, USA, June 27-30, 2016, pages 203–212. IEEE Computer Society.

[21] SimNet: Stepwise Image-Topic Merging Network for Generating Detailed and Comprehensive Image Captions. Fenglin Liu, Xuancheng Ren, Yuanxin Liu, Houfeng Wang and Xu Sun (2018). EMNLP.

[22] Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning. Jiasen Lu, Caiming Xiong, Devi Parikh, Richard Socher.(2017).CVPR.

[23] GroupCap: Group-Based Image Captioning With Structured Relevance and Diversity Constraints . Fuhai Chen ; Rongrong Ji ; Xiaoshuai Sun ; Yongjian Wu ; Jinsong Su (2018).CVPR.

[24] Towards Diverse and Natural Image Descriptions via a Conditional GAN. Bo Dai, Sanja Fidler et al. (2017). CVPR

[25] Speaking the Same Language: Matching Machine to Human Captions by Adversarial Training . Rakshith Shetty,Marcus Rohrbach, Lisa Anne Hendricks (2017) ICCV

[26] Attacking Visual Language Grounding with Adversarial Examples: A case Study on Neural Image Captioning . Hongge Chen, Huan Zhang, Pin-Yu Chen, Jinfeng Yi

[27] FOIL it! Find One mismatch between Image and Language caption
Ravi Shekhar, Sandro Pezzelle,Yauhen Klimovich

[28] Contrastive Learning for Image Captioning. Bo Dai, Dahua Lin

[29] Ranzato, Marc’Aurelio, Sumit Chopra, Michael Auli, and Wojciech Zaremba. “Sequence level training with recurrent neural networks.” （2016）ICLR

[30] Siqi Liu, Zhenhai Zhu, Ning Ye. Improved Image Captioning via Policy Gradient optimization of SPIDEr (2017) ICCV

[31] Zhou Ren, Xiaoyu Wang, Ning Zhang. Deep Reinforcement Learning-based Image Captioning with Embedding Reward (2017) CVPR

[32] Ramakanth Pasunuru and Mohit Bansal. Multi-Task Video Captioning with Video and Entailment Generation (2017) ACL

[33] Subhashini Venugopalan, Huijuan Xu, UMass Lowell, Jeff Donahue. Translating Videos to Natural Language Using Deep Recurrent Neural Networks. (2015) NAACL

[39] Du Tran, Lubomir Bourdev, Rob Fergus. Learning Spatiotemporal Features with 3D Convolutional Networks. (2015) CVPR

[40] Ramakanth Pasunuru, Mohit Bansal. Reinforced Video Captioning with Entailment Rewards (2017)EMNLP
