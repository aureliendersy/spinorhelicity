"""
Reference file for fast local numerical evaluations
This contains set of on-shell four momenta appropriate for numerical evaluations
We give two different sets of four-momenta for different n-pt amplitudes
For each set we additionally give the numerical evaluations of the different brackets terms

Momentas are generated using the S@M Package by Maître, D. and Mastrolia, P.
(http://dx.doi.org/10.1016/j.cpc.2008.05.002)
"""
import sympy as sp
from sympy import I

ZERO_ERROR_POW_LOCAL = 10

MOMENTA_DICT1 = {'4pt1': [2.168045478040097405141269641800320418259852629865936, -1.797278753194948589790282965177089448831570925395903, 0.431015921025208174042754337437197038185532025449617, -1.133329411065872350034886159092258676615480848915334],
                 '4pt2': [2.990889334140360719058390452405564500509081698352757, 0.950448532966487671904593215200594876251231996188987, -2.066013154487146162373749809597695824532348611161416, -1.942590085618294622838408461562373362346378054813246],
                 '4pt3': [-1.198250657185148825269965397737581178778650805528879, 1.077621204508127529151200442451245176447345110043617, -0.434073730662644751411937896172908891013057642504361, -0.293457276937729349711458246866888152393370714460853],
                 '4pt4': [-3.960684154995309298929694696468303739990283522689814, -0.230790984279666611265510692474750603867006180836701, 2.069070964124582739742933368333407677359874228216160, 3.369376773621896322584752867521520191355229618189434],
                 '4ptab12': -2.7533079397382581966117790791403609526372373853799 + 2.4864202500582103050280741239652513263008468883512 * I,
                 '4ptab13': -0.15599710095268670132815842082880162941774748438103 - 1.26047590242069504971758310420443844709544700723213 * I,
                 '4ptab14': -3.0628573034048373777868864276008902302641080672552 - 1.6639593786556223408124311212370744057553300610436 * I,
                 '4ptab23': 2.8284050627935309991473697336488675008560948244247 + 2.0371500866652606021692449602567309301639804319538 * I,
                 '4ptab24': -1.20327320962994116434825726447349961952211119291545 + 0.40653189183109354283712842215995397375139493104865 * I,
                 '4ptab34': 1.0450379696688465357383112340914030105620697868185 - 3.5596187875776361116408955203079461668827694937727 * I,
                 '4ptsb12': 2.7533079397382581966117790791403609526372373853799 + 2.4864202500582103050280741239652513263008468883512 * I,
                 '4ptsb13': -0.15599710095268670132815842082880162941774748438103 + 1.26047590242069504971758310420443844709544700723213 * I,
                 '4ptsb14': -3.0628573034048373777868864276008902302641080672552 + 1.6639593786556223408124311212370744057553300610436 * I,
                 '4ptsb23': 2.8284050627935309991473697336488675008560948244247 - 2.0371500866652606021692449602567309301639804319538 * I,
                 '4ptsb24': -1.20327320962994116434825726447349961952211119291545 - 0.40653189183109354283712842215995397375139493104865 * I,
                 '4ptsb34': -1.0450379696688465357383112340914030105620697868185 - 3.5596187875776361116408955203079461668827694937727 * I,
                 '5pt1': [0.802604371129375020513663450918090005214059752053782, -0.472467556767212874508196067131268579424598385492246, -0.584978759048806994730394505653263280964767918061610, -0.280620804325192260741823765840295846860670220508393],
                 '5pt2': [2.019074087960340448060501665277235567301789841702585,1.032144341119158052736087296781370686585041562677424, 1.567100579248108368800412801637883242869975691713759, -0.745341536672164084058600246695524145044300512534078],
                 '5pt3': [0.420671383355943675357382833679786541567778216075890,-0.2700160701953852038188994251891417153767154273830445, -0.164814883971256222317508720016244153422258701508239, -0.277293686607512729500584196420063006383931716150079],
                 '5pt4': [-1.598715179803680480635233461889066206362844227294077,-1.501243820939688313682318718561999488526395475918031,-0.001381493482949151272925509863715563463758161551449, 0.549686554047737308683752373359901277518530721999664],
                 '5pt5': [-1.643634662641978663296314487986045907720783582538181, 1.211583106783128339273326914101039096742667726115897, -0.815925442745096000479584066104660245019190910592460, 0.753569473557131765617255835595981720770371727192885],
                 '5ptab12': -1.3987834325177452843291803219472058117389678955078 -1.9169961235239592056700530520174652338027954101563*I,
                 '5ptab13': 0.26758239507867026540921528976468835026025772094727 +0.00788737865556470539585287582440287224017083644867*I,
                 '5ptab14': 0.8302624468316421868507859471719712018966674804688 -1.7287634816844124063806020785705186426639556884766*I,
                 '5ptab15': 1.3887143055965966986775583791313692927360534667969 +0.3108785113142938927310865437902975827455520629883*I,
                 '5ptab23': 1.1510916653001499465602819327614270150661468505859 +1.0170144888216816614345816560671664774417877197266*I,
                 '5ptab24': -1.4206474057933313748236514584277756512165069580078 -0.7175449717397238691418692724255379289388656616211*I,
                 '5ptab25': -0.3339271439966955412614879605825990438461303710938 +2.3121813392410719423253340210067108273506164550781*I,
                 '5ptab34': 0.44632021498564355965754657518118619918823242187500 -1.28537612837690939926460487185977399349212646484375*I,
                 '5ptab35': 0.73812166967382375482031875435495749115943908691406 -0.18648301892310431560062511380237992852926254272461*I,
                 '5ptab45': -2.6981627476837743984106054995208978652954101562500 +0.8845225022566414185831717986729927361011505126953 *I,
                 '5ptsb12': 1.3987834325177452843291803219472058117389678955078 -1.9169961235239592056700530520174652338027954101563*I,
                 '5ptsb13': -0.26758239507867026540921528976468835026025772094727 +0.00788737865556470539585287582440287224017083644867*I,
                 '5ptsb14': 0.8302624468316421868507859471719712018966674804688 +1.7287634816844124063806020785705186426639556884766*I,
                 '5ptsb15': 1.3887143055965966986775583791313692927360534667969 -0.3108785113142938927310865437902975827455520629883 *I,
                 '5ptsb23': -1.1510916653001499465602819327614270150661468505859 +1.0170144888216816614345816560671664774417877197266 *I,
                 '5ptsb24': -1.4206474057933313748236514584277756512165069580078 +0.7175449717397238691418692724255379289388656616211 *I,
                 '5ptsb25': -0.3339271439966955412614879605825990438461303710938 -2.3121813392410719423253340210067108273506164550781 *I,
                 '5ptsb34': 0.44632021498564355965754657518118619918823242187500 +1.28537612837690939926460487185977399349212646484375 *I,
                 '5ptsb35': 0.73812166967382371562740930970369485562786947504545 + 0.18648301892310430196943059834815820958266960646864*I,
                 '5ptsb45': 2.6981627476837743587924735360982936928095854456949 + 0.8845225022566413990762734797944109582000051364309*I,
                 '6pt1': [0.2579230105281449644764196620814336474739893030134599, 0.002899631300872412687362451171152691593336674396672, -0.134799942827913761432270269789805909480295219195104, 0.219874616342614138550445130694528799277357441731275],
                 '6pt2': [1.295689445917005072601243596903981530491445272395524, 0.345464874437348735319518845448250646787398966113121, 0.340629868094526125034665596469205566197322252790978, -1.201431002493529887324417637756800077323753692060379],
                 '6pt3': [2.955568007803374616697257803576720214584140717850992, -1.539501220156286676142453554663874062438293430632660, -1.089778417410148664403442187226763655930762328300285, -2.275456271351999637962470578422800847003542537387180],
                 '6pt4': [1.405684130859059361540489579754110409724754643683477, 0.102915788735810214934403975692175560412121154440233, -1.217108083761215121144959473393539039437420742260151, -0.695704052468417176199618013382391016022160062602463],
                 '6pt5': [-0.918443982340248394556060691407296967088984177578353, -0.263775941570350384350696756778009350576784808990499, -0.864076440534526554383654194493215200366117052710789, 0.165328479878308928268351039400613915525294756256273],
                 '6pt6': [-4.996420612767335620759349950908948835185345759365100, 1.351996867252605697551865039130304514222221444673133, 2.965133016439277976329660528434118239017273089675352, 3.787388230093023634667710059466849225546804094062474],
                 '6ptab12': -0.77650819957836791918026619896409101784229278564453 -0.82678289449041864056511030867113731801509857177734 *I,
                 '6ptab13': 1.2938224670408868810511648916872218251228332519531 +0.7525924561342149932130496381432749330997467041016 *I,
                 '6ptab14': -0.08089228561924499683044587072799913585186004638672 +0.83413385691047692294119997313828207552433013916016 *I,
                 '6ptab15': 0.85748371753261598549045174877392128109931945800781 -0.20645973893248739661210322537954198196530342102051 *I,
                 '6ptab16': -1.6495750204583694564774987156852148473262786865234 +0.8545338350513033143229790766781661659479141235352 *I,
                 '6ptab23': 1.5010958206256312941206942923599854111671447753906 +1.3206853585335205369943878395133651793003082275391 *I,
                 '6ptab24': 0.9106287789780505015357903175754472613334655761719 +1.3783300511087748230210081601398997008800506591797 *I,
                 '6ptab25': -0.65714817824024951420369689003564417362213134765625 +0.88318738972430832490090324427001178264617919921875 *I,
                 '6ptab26': -2.0478629066403728486989166412968188524246215820313 +1.6147649186632226925297572961426340043544769287109 *I,
                 '6ptab34': -1.6736707300005184695379512049839831888675689697266 +0.0777804835825084139866447685562889091670513153076 *I,
                 '6ptab35': 1.9679059197698829386524721485329791903495788574219 -1.8706866707399392169008933706209063529968261718750 *I,
                 '6ptab36': -0.77089572950127904782391397020546719431877136230469 -1.03860070352783284697295584919629618525505065917969 *I,
                 '6ptab45': 2.0925023149580348480469638161594048142433166503906 -0.1501144913223285692893682607973460108041763305664 *I,
                 '6ptab46': -0.68393654644346440274319616925022448100224557428112 + 1.17034839852315518987849495356645828936847077986776*I,
                 '6ptab56': -1.4012697500753135783462465826662426756440849527228 - 3.4350303286068863053222741264865031953266633091852*I,
                 '6ptsb12': 0.77650819957836791918026619896409101784229278564453 -0.82678289449041864056511030867113731801509857177734 *I,
                 '6ptsb13': -1.2938224670408868810511648916872218251228332519531 +0.7525924561342149932130496381432749330997467041016 *I,
                 '6ptsb14': 0.08089228561924499683044587072799913585186004638672 +0.83413385691047692294119997313828207552433013916016 *I,
                 '6ptsb15': 0.85748371753261598549045174877392128109931945800781 +0.20645973893248739661210322537954198196530342102051 *I,
                 '6ptsb16': -1.6495750204583694564774987156852148473262786865234 -0.8545338350513033143229790766781661659479141235352 *I,
                 '6ptsb23': -1.5010958206256312941206942923599854111671447753906 +1.3206853585335205369943878395133651793003082275391 *I,
                 '6ptsb24': -0.9106287789780505015357903175754472613334655761719 +1.3783300511087748230210081601398997008800506591797 *I,
                 '6ptsb25': -0.65714817824024951420369689003564417362213134765625 -0.88318738972430832490090324427001178264617919921875 *I,
                 '6ptsb26': -2.0478629066403728486989166412968188524246215820313 -1.6147649186632226925297572961426340043544769287109 *I,
                 '6ptsb34': 1.6736707300005184695379512049839831888675689697266 +0.0777804835825084139866447685562889091670513153076 *I,
                 '6ptsb35': 1.9679059197698829386524721485329791903495788574219 +1.8706866707399392169008933706209063529968261718750 *I,
                 '6ptsb36': -0.77089572950127904782391397020546719431877136230469 +1.03860070352783284697295584919629618525505065917969 *I,
                 '6ptsb45': 2.0925023149580348480469638161594048142433166503906 +0.1501144913223285692893682607973460108041763305664 *I,
                 '6ptsb46': -0.68393654644346440274319616925022448100224557428112 - 1.17034839852315518987849495356645828936847077986776*I,
                 '6ptsb56': 1.4012697500753135783462465826662426756440849527228 - 3.4350303286068863053222741264865031953266633091852*I
                 }

MOMENTA_DICT2 = {'4pt1': [0.574462717966232665019345509971475687048769179170527, 0.102214402405133375273918537472740122902384178547951, -0.514724220769173117507266786361929645626668725201582, -0.233706240455149960260986351316442783011782838325420],
                 '4pt2': [5.935888718956621147827693200668097197706294519666804, -5.776078109018630305866092084142676970397440362384601, 0.937569026266952349909030176487989442697219030071335, 0.996323684018943520510009428907177027437765309113243],
                 '4pt3': [-3.603693159699339828553590276566046054827207962324326, 3.434133887807194513901345639806506094469163997293404, -0.191364694282630693038832223884056141837798868980332, 1.075503781332343828013849438322951273184441437845074],
                 '4pt4': [-2.906658277223513984293448434073526829927855736513004, 2.239729818806302416690827906863430753025892186543247,-0.231480111215148539362931166242003655232751435889421, -1.838121224896137388262872515913685517610423908632898],
                 '4ptab12': 1.7416433563118277941134783388833089003278416077899 - 2.5294746086229563220746096655358830510896191794563 * I,
                 '4ptab13': 1.4722854241813297847720029145199284431086575904153 + 1.5391823953301393597228892698001885109941488758166 * I,
                 '4ptab14': 1.9827382421827458604131231271788801247905684351648 +0.9816341593936296251534643430243727248653358716760 *I,
                 '4ptab23': -0.2493243174330419424298873769853231703633873741964 +2.1983389057738369033347977325519792362509027201487 *I,
                 '4ptab24': -0.4958709808382689516330092214649384468976064926229 -2.0714291652657391041287996948819233485573388434692 *I,
                 '4ptab34': 3.0696708650521414108704004661161202237191184866444 -0.0931888217729156864041103322897624248362369499902 *I,
                 '4ptsb12': -1.7416433563118277941134783388833089003278416077899 -2.5294746086229563220746096655358830510896191794563 *I,
                 '4ptsb13': 1.4722854241813297847720029145199284431086575904153 -1.5391823953301393597228892698001885109941488758166 *I,
                 '4ptsb14': 1.9827382421827458604131231271788801247905684351648 -0.9816341593936296251534643430243727248653358716760*I,
                 '4ptsb23': -0.2493243174330419424298873769853231703633873741964 -2.1983389057738369033347977325519792362509027201487*I,
                 '4ptsb24': -0.4958709808382689516330092214649384468976064926229 +2.0714291652657391041287996948819233485573388434692 * I,
                 '4ptsb34': -3.0696708650521414108704004661161202237191184866444 - 0.0931888217729156864041103322897624248362369499902*I,
                 '5pt1': [1.4005554712036348244995907591000343708212005981465458, -1.267465562457767621848421538879664760290168466012080, -0.4011307840540465919044133474886368941799522205350894, -0.440659471685604620407304985507373867681692844985620],
                 '5pt2': [1.740930129470532407659177213314538767265612299488266, 1.003580852803893676621967054247268457414987751948905, 1.242702062786989125939081664357547374360789459176334, 0.692354512319127130371026125324647183065867191520369],
                 '5pt3': [1.0059123241914206362762326281004570191263797512901685, 0.462495084006385986261369601079839446399175219392784, -0.8713200262422717347853654296778219719466979447879305, -0.1968738507249720941593799932158108474645162844336695],
                 '5pt4': [-2.125777590184823835484260085989775791432891751927808, -1.442027049923951140369612718604435635011418973783768, -1.561324987016899536820918916917581736647831698504852, 0.041864485382025817654382985934044903131592802106050],
                 '5pt5': [-2.021620334680764032950740514525254365780300896997171, 1.243416675571439099334697602156992491487424468454158, 1.591073734526228737571616029726493228413692404651538, -0.096685675290576233458724132535507371051250864207129],
                 '5ptab12': -2.6483271193675586019367074186448007822036743164063 -1.4191784586191555916911966050975024700164794921875 *I,
                 '5ptab13': -1.6673859630695873068617629542131908237934112548828 +0.5808218551726263001455663470551371574401855468750 *I,
                 '5ptab14': 1.6506933863077146718012500059558078646659851074219 -2.8462052866232343539820703881559893488883972167969 *I,
                 '5ptab15': -0.47515285916049032621799597109202295541763305664063 -1.04584443255367243175157909718109294772148132324219 *I,
                 '5ptab23': -0.2233997108524329033851785197839490137994289398193 +2.2276516062559346131877191510284319519996643066406 *I,
                 '5ptab24': 0.53710233929600459390485411859117448329925537109375 -0.62948127954981047960103524019359610974788665771484 *I,
                 '5ptab25': -2.8647515087678425693695771769853308796882629394531 +2.2690342278338091475120563700329512357711791992188 *I,
                 '5ptab34': 2.3712373703806606251021094067255035042762756347656 -0.1562294961420575278854272482931264676153659820557 *I,
                 '5ptab35': 0.4266096714815090542773310882453751458133755270845 + 1.5168055092842691060378967113627092703306592797923*I,
                 '5ptab45': -2.6871592464389590845634132132678838871719082929203 - 3.1522608235701534108095194217033982513639963800413*I,
                 '5ptsb12': 2.6483271193675586019367074186448007822036743164063 -1.4191784586191555916911966050975024700164794921875 *I,
                 '5ptsb13': 1.6673859630695873068617629542131908237934112548828 +0.5808218551726263001455663470551371574401855468750 *I,
                 '5ptsb14': 1.6506933863077146718012500059558078646659851074219 +2.8462052866232343539820703881559893488883972167969 *I,
                 '5ptsb15': -0.47515285916049032621799597109202295541763305664063 +1.04584443255367243175157909718109294772148132324219 *I,
                 '5ptsb23': 0.2233997108524329033851785197839490137994289398193 +2.2276516062559346131877191510284319519996643066406 *I,
                 '5ptsb24': 0.53710233929600459390485411859117448329925537109375 +0.62948127954981047960103524019359610974788665771484 *I,
                 '5ptsb25': -2.8647515087678425693695771769853308796882629394531 -2.2690342278338091475120563700329512357711791992188 *I,
                 '5ptsb34': 2.3712373703806606251021094067255035042762756347656 +0.1562294961420575278854272482931264676153659820557 *I,
                 '5ptsb35': 0.4266096714815090542773310882453751458133755270845 - 1.5168055092842691060378967113627092703306592797923*I,
                 '5ptsb45': 2.6871592464389590845634132132678838871719082929203 - 3.1522608235701534108095194217033982513639963800413*I,
                 '6pt1': [0.853064352275053953745630552438112290437878850883607, -0.641126484008237941483390030180844246352892530479096, -0.4949466441810194791363360717095013478792054736308125, -0.267774980234597842767626966172729168434163348024076],
                 '6pt2': [2.676391426120099895287993210235526824969315803845844, 2.231268452779446097790061707625177766534674383817395, 1.058813539586362452765869105784335182260095827997192, 1.031225506777735461634478046504322023979555006654013],
                 '6pt3': [1.519650455432334465764965250753251545392278397975464, 1.164414124421205153696931793772069167585246561788544, -0.9764565900968324718654859387724277350427672989592365, -0.003127491097703322522191987386965061228376415427671],
                 '6pt4': [-2.043791855872309928294364909634565411112492178371441, -0.969671605853281122824384140628859001584471373551228, 0.0849903790291910568419593748725827086229432618876126, -1.797108444809153508602930408706058105012259601024219],
                 '6pt5': [-1.319937085644670986061563485757881707233000689489441, -0.117545570397955036690653286787595903011850615590592, 0.296748125297377861489186780140289430891335821625404, 1.280764419818281271241486150093688882712409680223921],
                 '6pt6': [-1.685377292310507400442660618034443542453980184844033, -1.667338916941177150488566043799947783170706425985023, 0.0308511903649205799048067496847217611475978610798400, -0.243979010454562058983214834332258572017165322401969],
                 '6ptab12': -2.5001593422613175299318299948936328291893005371094 -1.6664055004379245694678957079304382205009460449219 *I,
                 '6ptab13': -1.7553905792154871345900346568669192492961883544922 -0.1900891677323663098153616601848625577986240386963 *I,
                 '6ptab14': 1.2347362606813299112928916656528599560260772705078 -2.0209088458896165008127354667522013187408447265625 *I,
                 '6ptab15': -1.01900279772574764436399163969326764345169067382813 -0.62022309921167018131882286979816854000091552734375 *I,
                 '6ptab16': 0.8816343739752706687440308996883686631917953491211 -2.0823701028141869961984866677084937691688537597656 *I,
                 '6ptab23': -0.3936503217327744219566909578134072944521903991699 +2.2039461277134884120698643528157845139503479003906 *I,
                 '6ptab24': -1.1611796330423125311881449306383728981018066406250 +1.3183210256656023862120719059021212160587310791016 *I,
                 '6ptab25': -2.9958142446677986647785019158618524670600891113281 -0.9142198341415627149686429220309946686029434204102 *I,
                 '6ptab26': -0.80656500604909397988961927694617770612239837646484 -0.70177394413751803536882789558148942887783050537109 *I,
                 '6ptab34': 1.5005738577951905909912966308183968067169189453125 +1.2438006144838456012990945964702405035495758056641 *I,
                 '6ptab35': -1.6894441994380675620135434655821882188320159912109 -0.5442300571141177600864580199413467198610305786133 *I,
                 '6ptab36': 1.07402153520835086375484479503938928246498107910156 -0.16485417644044711438233719036361435428261756896973 *I,
                 '6ptab45': 1.0660152963642912649078198228380642831325531005859 -2.9298305674136395637674468162003904581069946289063 *I,
                 '6ptab46': 1.6652752450225930335775719963631075073353952274444 + 0.0167071095704525468197579818747712714066900911166*I,
                 '6ptab56': -0.5873576194183543306521072674808430182967205174124 + 2.0781885399115130875590698622652790679585536061221*I,
                 '6ptsb12': 2.5001593422613175299318299948936328291893005371094 -1.6664055004379245694678957079304382205009460449219 *I,
                 '6ptsb13': 1.7553905792154871345900346568669192492961883544922 -0.1900891677323663098153616601848625577986240386963 *I,
                 '6ptsb14': 1.2347362606813299112928916656528599560260772705078 +2.0209088458896165008127354667522013187408447265625 *I,
                 '6ptsb15': -1.01900279772574764436399163969326764345169067382813 +0.62022309921167018131882286979816854000091552734375 *I,
                 '6ptsb16': 0.8816343739752706687440308996883686631917953491211 +2.0823701028141869961984866677084937691688537597656 *I,
                 '6ptsb23': 0.3936503217327744219566909578134072944521903991699 +2.2039461277134884120698643528157845139503479003906 *I,
                 '6ptsb24': -1.1611796330423125311881449306383728981018066406250 -1.3183210256656023862120719059021212160587310791016 *I,
                 '6ptsb25': -2.9958142446677986647785019158618524670600891113281 +0.9142198341415627149686429220309946686029434204102 *I,
                 '6ptsb26': -0.80656500604909397988961927694617770612239837646484 +0.70177394413751803536882789558148942887783050537109 *I,
                 '6ptsb34': 1.5005738577951905909912966308183968067169189453125 -1.2438006144838456012990945964702405035495758056641 *I,
                 '6ptsb35': -1.6894441994380675620135434655821882188320159912109 +0.5442300571141177600864580199413467198610305786133 *I,
                 '6ptsb36': 1.07402153520835086375484479503938928246498107910156 +0.16485417644044711438233719036361435428261756896973 *I,
                 '6ptsb45': -1.0660152963642912649078198228380642831325531005859 -2.9298305674136395637674468162003904581069946289063 *I,
                 '6ptsb46': -1.6652752450225930335775719963631075073353952274444 + 0.0167071095704525468197579818747712714066900911166*I,
                 '6ptsb56': 0.5873576194183543306521072674808430182967205174124 + 2.0781885399115130875590698622652790679585536061221*I
                 }

MOMENTA_DICTS = [MOMENTA_DICT1, MOMENTA_DICT2]


def check_numerical_equiv_local(tokens, hypothesis, target, npt=None):
    """
    Given two sympy expressions we check numerically if they are equal
    :param tokens:
    :param hypothesis:
    :param target:
    :param npt
    :return:
    """
    # Check the n-pt of the expression to check if it is not given
    if npt is None:
        npt = max([int(str(symb)[-1]) for symb in hypothesis.free_symbols | target.free_symbols])

    # Add temporary check on the canonical ordering
    token_sp = [sp.parse_expr(tok) for tok in tokens if tok[-1] > tok[-2]]
    func_hyp = sp.lambdify(token_sp, hypothesis)
    func_tgt = sp.lambdify(token_sp, target)
    relevant_keys = ['{}pt'.format(npt)+tok for tok in tokens if tok[-1] > tok[-2]]

    rel_diff = 0
    for momenta_vals in MOMENTA_DICTS:
        relevant_coeffs = [momenta_vals[key] for key in relevant_keys]
        hyp_num = sp.N(func_hyp(*relevant_coeffs), ZERO_ERROR_POW_LOCAL+5)
        tgt_num = sp.N(func_tgt(*relevant_coeffs), ZERO_ERROR_POW_LOCAL+5)
        diff = abs(tgt_num - hyp_num)

        # If the target is close to 0 then we simply add the difference instead of the relative difference
        if abs(tgt_num) < 10 ** (-ZERO_ERROR_POW_LOCAL):
            rel_diff += float(diff)
        else:
            rel_diff += float(diff/abs(tgt_num))

    valid = rel_diff < 10 ** (-ZERO_ERROR_POW_LOCAL)
    return valid, rel_diff
