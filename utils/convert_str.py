"""
    字符串转换,分段符号替换成\\n，方便在json中使用，用于手动构建json格式剧本
"""


def convert_newlines_to_literal(input_string):
    # 将换行符替换为其字面表示 "\\n"
    return input_string.replace("\n", "\\n").replace("\"", "\\\"")



multiline_string = """庄清纯
你叫庄清纯，是一个漂亮可爱的女孩。2007年，你的父亲出轨，父母离婚。2009年，母亲带着九岁的你嫁进了秦家。
你的继父叫秦专家，是无人不知无人不晓的音乐家。他有一个儿子叫秦老大，与你同岁。受你继父的影响，你开始走上音乐的道路，极有天赋的你很快就小有成就。
2014年的一天，你偶然在秦专家的电脑上发现了一个文件，里面是几张照片，是你生父出轨的证据。你往下拉，发现是一组几乎一样的照片，但是里面男人的脸却不是你生父的脸。
你明白了，所谓的父亲出轨导致的离婚，其实都是秦专家的杰作，目的只是让母亲离开父亲嫁给他。其实在之前，你早就听说过继父在外面拈花惹草的事情，但都以为是谣言并没有在意，现在看来，件件属实。
你心里顿时充满了对秦专家的憎恨。你把这件事情告诉了母亲，但不料她贪恋秦专家的钱财和名气，不愿意再离婚了。你无奈于母亲的贪欲，只能自己孤身与秦专家做对。
不久后，你初中毕业了。秦专家作为最有名的音乐院校院长，本来想安排你进入附属高中就读，但你因为厌恶秦专家所以坚决不答应。所以，你自作主张报考了清北高中。
2016年9月1日，读高二的你作为艺术生被分到了高二（7）班。这是一个文科班，班里以女生和术科生居多。开学第一节班会课的时候，你惊讶地发现你异父异母的哥哥秦老大也跟你一个班。你觉得有其父必有其子，所以你对于秦老大的厌恶不亚于秦专家。你警告秦老大，让他跟自己保持距离，如果给别人发现了你们的兄妹关系，你就对他不客气。他的身边总跟着郑友好和韦小弟，你也十分看不起他们，而且韦小弟好像总是有意无意地盯着你。
在班里，你认识了一个女生，叫做贾美丽。你们同为音乐生，而且性情相投，很快就成为了无话不谈的好闺蜜。但即使是这样的朋友，你也不曾跟她透漏过你与秦老大的关系。
2017年初夏，你有一次路过学校篮球场时，被飞来的篮球重重砸中了头部。失手砸你的人叫萧体育，是你们班的体育老师，高大帅气的他让不少女同学犯过花痴，你也不例外。
萧体育觉得过意不去，就送你到医务室检查了一下，又帮你买了跌打损伤的药，送你回了宿舍。你觉得萧体育非常温柔体贴，对他有了好感。你知道他已经有了女朋友，还是你们班的音乐老师，叫做薛媚娘，美丽而有气质。但你还是决定追求他，也算是证明自己魅力的一种方式吧。
从那以后，你就经常去看萧体育打篮球，还给他送水送手巾。他似乎感觉到了你的爱意，刻意与你保持距离。但你是个倔强的人，萧体育越躲避，你就越来劲，似乎不得到他的回应不罢休。你的执着感动了萧体育，他违反学校规定答应了你的追求，与你成为了一对地下情人。
2018年元旦刚过，你的母亲就被查出了胃癌中期，需要住院治疗。这个病有痊愈的可能，但是需要巨额医药费。好在秦专家经济实力雄厚，你并不担心钱的问题。
2018年2月14日是情人节，也是你的生日。在之前，你曾经跟萧体育说，你很喜欢“庄周梦蝶”的故事，一直渴望自己就是那个庄氏文人，在某天在梦中幻化为栩栩如生的蝴蝶。所以，萧体育特地到当地最有名的首饰作坊亲手为你设计打造了一款独一无二蝴蝶耳钉，作为情人节礼物和生日礼物。你非常感动，对这款耳钉爱不释手，每一天都戴着。
2018年3月，你觉得贾美丽好像一夜之间变得非常有钱，常常换奢侈品包包和衣服，而这些都是她以前买不起的。你问她原因，她总是含糊其辞、转移话题。
2018年4月底，你和萧体育在体育器材室偷偷亲热的时候，被贾美丽撞见了。萧体育问你该怎么办，你说没事的，你能处理好。
到了晚上，你找到贾美丽，请她不要把今天的事情告诉别人，特别是薛媚娘。你本以为她作为闺蜜不会为难你，没想到她却狮子大开口，让你给她一笔封口费，数目还不小。你突然觉得她变得很陌生，那次的谈话不欢而散。之后贾美丽也时不时地跟你要钱，但是你们每次都吵得很厉害，贾美丽还经常在同学们面前提到你和萧体育，语气暧昧，由于你及时岔开话题，并没有引起同学们的怀疑。还好没多久贾美丽就停止了这种行为，你猜想是贾美丽念及往日情分或者有其他更重要的事情要处理，便松了口气。
2018年5月初，你的母亲突然病情急剧恶化，不久后就去世了。其实在这之前，母亲的病已经有了很大程度的好转，为什么会突然之间恶化，你想不通。
母亲去世的第二天，你到医院收拾病房里的遗物。途径医生办公室的时候，你看到秦专家和母亲的主治医师在谈话，还隐约听到了“怎么这么严重”、“当初不应该换她的药啊”什么的。你没有在意，继续往病房走去。
当你收拾到一半的时候，一个护士拿着母亲住院期间的医疗费用单走了进来。你让护士把单子给你，她却说秦专家嘱咐过这个单子只能亲手交给他。你有点生气，自己母亲的单据竟然不能交给你，于是就从护士的手里把单子抢了过来。
你注意到，从2018年3月底开始，母亲每天都要用的天价特效药换成了廉价的三等药。你觉得母亲的死不是单纯的病灾，而是人祸。你决定调查清楚，以慰藉母亲的在天之灵。
2018年5月15日，你联系了当地有名的真相大白私家侦探事务所，请求帮助。2018年5月25日，侦探所给你发了邮件，里面有你要的真相。
原来，贾美丽为了考上最好的音乐院校，勾引秦专家并在2018年1月与他发生了关系。2018年3月，贾美丽查出自己意外怀孕，向秦专家索要巨额赔偿1000万，不然就将这桩丑事昭告天下，让他身败名裂。秦专家为了保住自己的名誉，偷偷把母亲的救命药换成廉价药，还找关系偷偷转移了母亲名下的一套房产，抵押给银行换取贷款。
这一切让你对两个人恨之入骨，你决定解决掉这对狗男女，为母亲报仇。经过考虑，你决定先对贾美丽下手。
2018年5月31日，你准备了一顶男士假发、一套黑色男款运动服、一双男士休闲鞋和一根绳子，并把它们藏到了宿舍楼旁的小树林里。
【2018年6月1日】
2018年6月1日是高考前的最后一个星期五，全校的考生在这一天结束后都会回家休息、待战高考。在这一天晚上，你们高三（7）的同学都会到学校附近的餐厅吃散伙饭。你准备在饭后动手。
19:00，散伙饭准时开宴。席间，平时跟秦老大并无交集的贾美丽对秦老大特别殷勤，还主动敬酒。
21:00，大家吃完了饭，准备各回各家。你发现秦老大精神有点萎靡，以为是喝多了。贾美丽说她家比较远，打算第二天再回。你为了方便动手，便说你也要留宿，与贾美丽一起走回了学校。
21:05，你们二人走到了学校门口。贾美丽说她要到琴房拿琴谱，让你自己先回去。这正合你意，于是你们就分开了。你来到了藏有衣物和绳子的小树林，快速换装后带着绳子向宿舍走去。
21:20，你进入宿舍楼，来到了贾美丽宿舍，躲进厕所里，等她回来就动手。因为是曾经的闺蜜，所以你们都有彼此宿舍的钥匙。
21:25，贾美丽血迹斑斑地回来了。你趁她不备从背后勒住了她。她似乎受了伤，并无力气挣扎，你轻而易举就将她勒死了。
21:30，你避开所有的血迹离开贾美丽的宿舍，来到小树林换下作案的衣服并藏好。
21:40，你回到自己的宿舍。
22:30，你收到了警察的传唤，你以为贾美丽的死已经被人发现了，没想到的是，秦老大竟然也惨死宿舍之中。
很显然，你就是杀死贾美丽的凶手，本案的真凶，祝你好运~但是好奇怪，秦老大又是因何而死呢？

"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """
『支线任务』：
①隐藏你和秦老大的兄妹关系。（10分）
②找出韦小弟家里的秘密。（20分）
『提示』：
中间女扮男装行凶的过程你的时间线是空白的，建议不要太早编造自己去做了其他事的谎言，可能会有目击证人喔~
『注意』：
玩家不能公开自己的支线任务，否则得分不计。
"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """萧体育
你叫萧体育，篮球运动员出身，2010年退役后到清北高中教体育，2014年担任该校体育部主任。
身材高大、阳光帅气的你一直是女教师们仰慕的对象，但这么多年过去了，你始终没有找到另一半。也许缘分未到，你总跟自己这么说。
2016年9月，学校来了一个新的音乐老师，叫做薛媚娘。这位女老师年纪不小，但却极具魅力，身上独特的气质把你一下子吸引住了。你决定对她展开爱情攻势，不久后她答应了你的追求，跟你在一起了。你们的爱情还算甜蜜，是学校里人人羡慕的金童玉女。
2017年初夏，你与学生们一起打篮球，不小心砸中了场边的一个女生。这个女生你认识，是你们班上的同学，叫做庄清纯。你觉得过意不去，就送她到医务室检查了一下，又帮她买了跌打损伤的药，送她回了宿舍。
从那以后，庄清纯就经常去看你打篮球，还给你送水送手巾。你感觉到了她的爱意，但是你已经有女友了，就刻意与庄清纯保持距离。
没想到你越躲避，庄清纯就越来劲，似乎不得到你的回应不罢休。就在这个时候，你感觉到薛媚娘似乎对你越来越冷淡。你没能守住防线，违反学校规定与庄清纯成了一对地下情人。
2018年2月14日是情人节，也是庄清纯的生日。在之前庄清纯跟你说，她很喜欢“庄周梦蝶”的故事，一直渴望自己就是那个庄氏文人，在某天在梦中幻化为栩栩如生的蝴蝶。因此，你特地到当地最有名的首饰作坊亲手为庄清纯设计打造了一款独一无二蝴蝶耳钉。此后，她就一直带着这对耳钉，喜欢得不得了。
2018年4月底，你和庄清纯在体育器材室偷偷亲热的时候，被贾美丽撞见了。这个贾美丽也是你教的学生，还是庄清纯的闺蜜。你问庄清纯怎么办，她说没事的，她能处理好。你心里还是不安，这毕竟关系到你的名声，但过了两天发现并没有什么风言风语，以为庄清纯已经妥善解决了，才放下心来。
谁知大概过了一周，贾美丽就来找你，要一大笔封口费，但你根本拿不出那么多钱。贾美丽威胁你说如果你不给她这笔钱，她就把这件事告诉校领导，让你身败名裂。你想到自己这么多年混到这个领导位置上有多不容易，而且如果被发现自己跟学生谈恋爱，怕是以后找工作都很难了，但是这笔钱又是自己无论如何都不可能凑到的。于是你们争了很久，不欢而散。
没想到第二天教务处领导就召开了会议，说有学生举报某些学校领导有作风问题，会议上你收到了贾美丽的短信：“再不给钱下次举报的时候我就不会只说某些校领导了！”。你知道贾美丽不拿到钱是不会善罢甘休的，便打算一不做二不休，让她带着这个秘密永远地闭嘴！
于是你找她商量了很久，说自己实在没办法这么快筹到那么多钱，让她给自己一个月的时间。贾美丽答应了你，但这期间你并没有筹钱，而是在网上购买了一瓶“心力交瘁粉”，服用数小时后会让人心脏功能衰竭，准备伪装成贾美丽应考压力太大猝死的样子。
【2018年6月1日】
2018年6月1日是高考前的最后一个星期五，全校的考生在这一天结束后都会回家休息、待战高考。这天晚上，你是值班的领导，负责巡逻校园里的几栋大楼。你准备今晚给贾美丽下毒，所以正常时间21：30下班的你延长了你的巡逻时间。这样就算查出贾美丽不是猝死而是中毒死的，也让大家以为贾美丽是在散伙饭上中的毒。
20:30，你开始巡逻校园的区域。
21:30，你才进入学生宿舍楼开始巡逻，想伺机寻找贾美丽。
21:37，你巡逻到了高三（7）班女生宿舍的楼层，发现其中一个宿舍门前的灯泡烧了，便顺手换上了新灯泡。却发现门外有血迹，而且居然正好是贾美丽的宿舍。你用巡逻钥匙打开了房门，发现贾美丽死绝在宿舍地上。你暗地庆幸自己不用动手了，正准备报警，却发现地上有一个晃眼的东西，你捡起来一看，发现是你设计打造的那款蝴蝶耳钉。这款耳钉只有庄清纯有，而且两人并不在一个宿舍。你怀疑这起案子与庄清纯脱不了干系。你不忍心让你的小情人身陷牢狱之灾，就决定先替她隐瞒下来，过后再向她询问情况。你藏起耳钉，清理干净贾美丽宿舍外面的血迹。
21:45，你离开了贾美丽宿舍，前去男生宿舍继续巡逻。
22:00，你巡逻到了高三（7）男生宿舍的楼层，发现又一个宿舍门前有血迹。你刚刚打开宿舍门、开了灯，一个中年男子就走了进来并大叫了一声。你回头一看，发现秦老大血肉模糊的尸体。后来才知道那是秦老大的司机，你们一起报了警。
22:30，你接到了警察的传唤。
"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """『支线任务』：
①不要那么快就暴露自己知道贾美丽死亡的事情，并且隐藏你清理贾美丽死亡现场的行为。（10分）
②找到薛媚娘的秘密。（20分）
『提示』：
虽然你很明显不是凶手，但你准备投毒的“心力交瘁粉”还没解决掉，这可能很危险喔~
『注意』：
玩家不能公开自己的支线任务，否则得分不计。
"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()


multiline_string = """"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()

multiline_string = """"""
converted_string = convert_newlines_to_literal(multiline_string)
print(converted_string)
print()
print()
