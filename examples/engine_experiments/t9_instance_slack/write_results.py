import json,math,re,hashlib
from pathlib import Path
from fractions import Fraction as F
from experiment import LANE,DATA
A=json.loads((DATA/'ANALYSIS.json').read_text());H=A['splits']['holdout'];DEV=A['splits']['development'];audit=json.loads((DATA/'EXACT_AUDIT.json').read_text())
load=lambda n:json.loads((DATA/(n+'.json')).read_text())
base={r['name']:r for r in load('F1')['rows']};strong={r['name']:r for r in load('F1_SUCCESSORS')['rows']};cond={r['name']:r for r in load('PORT_CONDITIONED_QUOTIENT')['rows']};quot={r['name']:r for r in load('DISTANCE_QUOTIENT')['rows']};router={r['name']:r for r in load('F1_ROUTER_REPLAY')['rows']}
fmt=lambda x: '—' if x is None else f'{x:.6g}'
lines=[]
def put(s=''):
 lines.append(s)
 if s.startswith('#'):lines.append('')
put('# T9 — gratis täckningsstatistik bär på små grafer; F1-förlusten sitter främst i potentialen')
put('2026-10-01. Form B. 400 grafer, 200 utveckling + 200 färska holdout; fyra riktiga F1-frågor med fulla exakta rationella referenser. Ingen källgraf, T1 eller T7 ändrad.')
put('')
put('**G1 klarar det angivna fällkriteriet genom kantantal: holdout ρ=0,5392 vid 128 lokala uppdateringar. Hubbandel når bara −0,3613. G3 vinner på små holdoutgrafer men faller vid överföring till F1. G2:s hubbundvikande flöde hjälper ett av tre svåra F1-fall. Den bytta representationen — en portkonditionerad avståndskvotient för potentialen — tar 975,906 → 1,131 på fullbanken vid samma 128 lokala uppdateringar. Mot starkare CG12 är det 2,336 → 0,919; det är ingen 10×-vinst mot bästa CG. F1:s första teckenbeslut på de fyra prövade trösklarna ändras inte.**')
put('')
put('## Förmåga → hinder → operation')
put('Förmågan är användbara, verifierbara intervall och strikta tröskeltecken utan full återvinning av originalgrafens potential. Det bindande hindret måste delas: U/R mäter flödets förlust och R/L potentialens förlust. På fullbanken vid start är U/R=1,8240 men R/L=777,862. Att huvudsakligen skylla på shortest-path-flödet genom hubben beskriver alltså fel faktor i just detta fall.')
put('')
put('SP behåller exakt T1:s initialization/schedule. ED tar bort högsta grads icke-portnod, packar högst åtta greedy kantdisjunkta shortest-resistance-vägar och delar ström med inversa väg-resistanser. Ingen väg ger SP-fallback. Detta är **inte** exakt Menger-packning. Potential och efterföljande cykel-/koordinatordning är oförändrade. Retained-SP testas som säker startup-kontroll; den kan inte få större U vid budget noll.')
put('')
put('Efter små/negativa utfall kördes bladpruning med exakt harmonic extension, två hubbkoordinater, trädinducerad potential, avståndskvotient och slutligen portkonditionering. De tre första tar inte bort F1-barriären; trädpotential blir sämre genom korskanternas energi. Avståndskvotienten minimerar potentialenergin i ett rum konstant på redan beräknade sink-distance-klasser. Portkonditionering gör båda frågeportarna till singleton-klasser. Flöde, träd, uppdateringsordning och budget är oförändrade. Den extra aggregation/små eliminering som bygger vittnet betalas separat; kvotienten är **inte en gratis statistik**.')
put('')
put('## Bevis och exakta kontrollkvitton')
put('För Bf=e_s−e_t och φ_s−φ_t=1: U=Σf²/c, D=Σc(Δφ)², L=1/D. Thomson/Dirichlet och exakt Cauchy-Schwarz ger L≤R≤U. Identiteten 1+(U−L)/L=(U/R)(R/L) separerar de två förlustfaktorerna.')
put('')
put('Den ursprungliga distance-potentialen ligger i kvotientens rum. Exakt Dirichlet-minimering ger D_q≤D_distance och därmed L_q≥L_distance. Singleton-portar utvidgar rummet, så D_cond≤D_q och L_cond≥L_q. Garanterad startup-dominans verifieras också i tester. Detta är klassisk variations-/Galerkin-algebra, ingen ny sats om Laplacianlösare. Kvotienten är cap:ad vid 64 klasser; på små grafer med singleton-klasser kan den sammanfalla med full lösning. På F1 löstes bara 12–22 okända, mot 3457–6902 i facit.')
put('')
put(f'**{audit["endpoint_rows_checked"]} serialiserade endpoint-rader och {audit["recorded_trajectory_containment_checks"]} registrerade rationella bankontroller: 0 inneslutningsbrott.** Alla energier och conservation constraints verifieras med Fraction på originalkanterna. Varje F1-ref bygger på oberoende exakt star-mesh-eliminering och exakt återkontroll av Kirchhoffs lag på samtliga originalkanter. Python-flint fmpq accelererar endast facit; kandidaten/checkern förblir Python Fraction. 20 facit matchar separat Fraction-Gauss, och fullbankens flint-ref matchar Fraction-piloten exakt.')
put('')
put('Sluttester: **2072 passed, 49,55 s**, se [delivery_tests.log](delivery_tests.log). Default-vittnen jämförs fält för fält mot bevarad T1, ingen default-ändring. 200 native router-replays matchar exakt alla valda endpoints, uppdaterings- och first-sign-antal; även alla fyra F1 native replays matchar valda enkelformer. [EXACT_AUDIT.json]('+str(DATA/'EXACT_AUDIT.json')+') listar råfiler och hashkvitton.')
put('')
put('## G1 — samtliga prövade statistikor')
put('Primärt gap g=(U−L)/L, samma definition som T1 LARGE:s citerade F1-tal. Sekundärt g_R=(U−L)/R_exakt. Budgeten är 128 T1-lokala attempts; lokalt attempt jämförs aldrig med en CG-iteration. Budgettak och faktiska attempts finns i rådata; exakt färdiga banor behöver inte betala meningslösa ytterligare steg.')
put('')
put('| Statistik | ρ utveckling, g | ρ holdout, g | ρ holdout, g_R |')
put('|---|---:|---:|---:|')
for k in H['correlations']:
 put(f'| {k} | {fmt(DEV["correlations"][k]["128"]["gap_lo"])} | {fmt(H["correlations"][k]["128"]["gap_lo"])} | {fmt(H["correlations"][k]["128"]["gap_exact"])} |')
put('')
put('Kantantal och komponentstorlek är redan kända efter obligatorisk edge scan; routergraden läses från byggd incidenslista. Hela extrastatistikpaketet kostade i median 5,99 % av det använda Fraction-facits tid. Menger och spektralgap kostade separat 35,0 % respektive 13,4 % och används **inte** som gratis routersignaler här. Initial_certificate_gap är en certifikatdiagnostik, inte oberoende strukturbevis. Path_hub_exposure var förregistrerad men kompletterades efter routerfreeze och används inte för att välja om modellen. Exakt resistansavstånd/diameter är avvisade som gratis feature: facit/all-pairs-arbete får inte läcka in i router.')
put('')
put('Familjer: path, cycle, grid, star, hub+ring, preferential attachment, sparse random, dense, weak bridge/barbell och parallella alternativ. 20 friska seeds per familj och split, n≈12–36, rationella vikter, slumpade portar/relabelling/kantordning. M-beroendet beskriver bl.a. täckning av ett fast lokalt schema; detta är inte en allmän lag för lika stora grafer. Familjestrata och ρ vid 0/32/128/512 finns i [ANALYSIS.json]('+str(DATA/'ANALYSIS.json')+'); konstanta stratas ρ är null, inte noll. F1:s två UNGROUNDED-frågor har samma globala gradstatistik men olika portar och helt olika gap, ett direkt hinder för en global-grad-router.')
put('')
put('## G2 — samma budget på F1, fulla exakta referenser')
put('F1-bankens relationer saknar fysikaliska konduktanser: deklarerad enhetskonduktans per undirected relation, parallellrader kvar. J3:s enstaka 10⁻¹⁸-float tolkas som exakt representerat binärvärde, samma som T1. Resultatet gäller denna strukturella modell, inte vetenskapliga/biologiska relationers styrka.')
put('')
put('| Fall | Exakt R, visad approximation | Facit s | Elimineringar | max elim-grad |')
put('|---|---:|---:|---:|---:|')
for n,r in base.items():
 q=r['reference_receipt'];put(f'| {n} | {fmt(float(F(r["reference"])))} | {q["wall_seconds"]:.3f} | {q["eliminations"]} | {q["max_elimination_degree"]} |')
put('')
put('Fullbanken har 4679 registrerade noder; frågekomponenten har 3459. Glued-fallen behåller två ägares originalkanter och precis de portar T1 använde. Full source/hash/portdata finns i rådata. Q1 har R≈10¹⁸ från svag brygga; startens relativa gap ≈5,01·10⁻¹⁴ har absolut bredd cirka 50 096, inte en exakt liten absolut resistansförlust.')
put('')
put('| 128 lokala attempts | SP | ED | fryst router | portkond. SP | portkond. ED | bäst gammal / bäst portkond. |')
put('|---|---:|---:|---:|---:|---:|---:|')
for n,r in base.items():
 vals=[r['forms'][k]['points']['128']['gap_lo'] for k in ['shortest','edge_disjoint']]
 pc=[cond[n]['forms']['quotient_cycle_'+k]['points']['128']['gap_lo'] for k in ['shortest','edge_disjoint']]
 rr=router[n]['forms']['cycle']['points']['128']['gap_lo'];ratio=min(vals)/min(pc) if min(pc)>0 else math.inf
 put('| '+n+' | '+' | '.join(fmt(v) for v in [*vals,rr,*pc])+' | '+('— (svag-brygge-nämnare)' if n.endswith('Q1') else fmt(ratio)+'×')+' |')
put('')
put('| 12 CG-iterationer | SP | ED | fryst router | portkond. SP | portkond. ED |')
put('|---|---:|---:|---:|---:|---:|')
for n in base:
 vals=[strong[n]['forms']['cg_'+k]['points']['12']['gap_lo'] for k in ['shortest','edge_disjoint']]
 pc=[cond[n]['forms']['quotient_cg_'+k]['points']['12']['gap_lo'] for k in ['shortest','edge_disjoint']]
 rr=router[n]['forms']['cg']['points']['12']['gap_lo'];put('| '+n+' | '+' | '.join(fmt(v) for v in [*vals,rr,*pc])+' |')
put('')
put('Hub-undvikande flöde förbättrar FULL3459 Q0 från 9,29407 till 5,08208 vid exakt samma CG12, men ändrar inte fullbank/UNGROUNDED Q0. En allmän hubbhypotes för G2 bär därför inte; ED är heller inte samma totala startarbete som en väg. Avståndskvotienten utan singleton-portar gav redan fullbankens gap 1,74029/1,46770 vid cycle128/CG12; portkonditionering förbättrade till 1,13135/0,919341. Ungrounded Q0 förbättras kraftigt lokalt men inte vid CG12. Samtliga misslyckade varmstarter ligger kvar i F1_SUCCESSORS.json/TREE_VOLTAGE.json.')
put('')
put('## G3 — liten holdoutvinst, misslyckad F1-transfer')
put('Routern väljs på 200 utvecklingsgrafer, frys före holdout: **maxgrad/medelgrad ≤3,245 → ED, annars SP**. Mått: medel log(1+g), lägre bättre. Ingen facitinput i feature eller val. Native funktionen beräknar bara vald form, båda formerna produceras inte i fråga.')
put('')
put('| Budget | SP, mean log1p(g) | ED | router |')
put('|---|---:|---:|---:|')
for b,v in H['budgets'].items():put('| '+b+' | '+' | '.join(fmt(v[k]['mean_log1p_gap']) for k in ['shortest','edge_disjoint','routed'])+' |')
put('')
v=H['budgets']['128'];d=v['routed']['difference_vs_edge_disjoint'];pct=(v['edge_disjoint']['mean_log1p_gap']-v['routed']['mean_log1p_gap'])/v['edge_disjoint']['mean_log1p_gap']*100
put(f'Vid 128: router {pct:.2f} % bättre än bästa enkelform ED; parad bootstrap 95 %-intervall för differensen router−ED [{d["bootstrap_95_ci"][0]:.6f}, {d["bootstrap_95_ci"][1]:.6f}], 19 förbättrade och 10 försämrade grafer. Median g SP/ED/router = {v["shortest"]["median_gap"]:.6f}/{v["edge_disjoint"]["median_gap"]:.6f}/{v["routed"]["median_gap"]:.6f}; p90 = {v["shortest"]["p90_gap"]:.6f}/{v["edge_disjoint"]["p90_gap"]:.6f}/{v["routed"]["p90_gap"]:.6f}. Detta är en liten vinst, inte 10×.')
put('')
put('F1-replay väljer SP i samtliga fyra frågor (skew 41,91–83,72). Medel log1p(g) vid cycle128: SP/router 5,344689, ED 5,030103; vid CG12: SP/router 1,177196, ED 1,045641. **G3 klarar sitt holdout-fällkriterium men faller som F1-router.** Kvotientoperationerna är separat efterföljande mekanismmätning på fyra F1-frågor; de har inte stoppats in i en efterhandsoptimerad holdout-router.')
put('')
put('## G4 — uppdateringar till tecken, med censur')
put('Strict sign ges enbart av L>theta eller U<theta. Theta/R ∈ {0,8;0,95;1,05;1,2}, exakt rationella diagnostiska trösklar. Tak 512 attempts; censurer kodas 513 endast för **ordinal** Spearman/median nedan, aldrig som uppmätt lösningstid.')
put('')
put('| Holdout, theta/R | SP avgjorda /200 | ED | router | median ordinal SP/ED/router |')
put('|---|---:|---:|---:|---|')
for q in ['4/5','19/20','21/20','6/5']:
 r=[H['cost'][k]['first_sign'][q] for k in ['shortest','edge_disjoint','routed']]
 put(f'| {float(F(q)):.2f} | {r[0]["decided"]} | {r[1]["decided"]} | {r[2]["decided"]} | '+ '/'.join(fmt(t['median_censored_ordinal']) for t in r)+' |')
put('')
put('| Statistik, SP first sign | ρ vid 0,8R | ρ vid 0,95R | ρ vid 1,05R | ρ vid 1,2R |')
put('|---|---:|---:|---:|---:|')
for k,r in H['correlations'].items():put('| '+k+' | '+' | '.join(fmt(r['first_sign'][q]['rho_censored_ordinal']) for q in ['4/5','19/20','21/20','6/5'])+' |')
put('')
put('Antal censurer i SP: 21/68/22/15 per tröskel ovan. Kantantal förutsäger översidans first-sign-antal starkt (ρ=0,734/0,742), men undersidan svagt (0,241/0,230); n når 0,537 vid 0,95R. Hubbandel når inte 0,5 på någon sida. En gratis statistik för gap är alltså inte automatiskt samma statistik för båda riktningars tecken.')
put('')
put('På F1 ändrar portkonditionerad potential **inte** first-sign-antal mot samma flödesforms CG-kontroll: fullbanken certifierar 1,2R vid iteration 9; de tre närmare/nedre trösklarna är censurerade vid 12. FULL3459 Q0 och UNGROUNDED Q0 avgör inga av dessa fyra vid CG12; Q1 avgör samtliga vid initialization. Även cycle512 för de tre svåra fallen är censurerad för alla fyra. Stor gapreduktion gav därmed inget ytterligare beslut vid de faktiskt prövade nära trösklarna. Ingen native margin_net/decision_cert-workload eller semantisk motsägelsevinst har påvisats; real theta-kontrakt kvarstår som saknad indata.')
put('')
put('Körd operation med tydlig teckeneffekt: stjärnor n=32/256/4096 har exakt R=2 och optimal SP-U=2; hubben är nödvändig. Dåliga start-L ger g=14,5/126,5/2046,5. Bladpruning + full original-graph extension ger L=U=2 vid **0** attempts och avgör alla fyra trösklar vid 0; SP:s undersida är fortfarande censurerad vid 512. n4096 replay av produktion+pruning+full verifiering vid budget0 tog 29,46 ms. Det är en klassisk strukturkontroll, inte F1:s fulla lösning. Portkonditionerad kvotient löser också stjärnan exakt i tester.')
put('')
put('## Kontroll och hela kostnaden')
put('Setup beror på vittnesform: ED flera Dijkstra-pass, varmhubben två extra koordinater, kvotienten en extra edge aggregation + liten exact solve. De får aldrig kallas gratis för att max_updates är lika. Följande mätningar inkluderar start, exakt verifiering och forskningsinstrumentens facitjämförelser från förberedda kanttriplar. Källläsning, gluing, hash och serialisering ingår i respektive jobbs totalkostnad nedan. Tider är receipt på delad maskin, inga kontrollerade latens-/speedup-påståenden.')
put('')
put('| F1, samma CG12 | SP kontroll s | ED kontroll s | portkond. SP s | portkond. ED s | k / okända | extra kvotient SP ms | elim multiply-subtracts |')
put('|---|---:|---:|---:|---:|---:|---:|---:|')
for n in base:
 c=cond[n]['forms'];p=c['quotient_cg_shortest'];put('| '+n+' | '+' | '.join(f'{x:.3f}' for x in [strong[n]['forms']['cg_shortest']['points']['12']['elapsed_seconds'],strong[n]['forms']['cg_edge_disjoint']['points']['12']['elapsed_seconds'],p['points']['12']['elapsed_seconds'],c['quotient_cg_edge_disjoint']['points']['12']['elapsed_seconds']])+f' | {p["quotient_groups"]} / {p["quotient_unknowns"]} | {p["quotient_seconds"]*1000:.2f} | {p["quotient_elimination_updates"]} |')
put('')
native=load('ROUTER_REPLAY')['rows'];native128=sum(r['routed']['points']['128']['elapsed_seconds'] for r in native)
put(f'Holdout200, summerad kostnad vid 128: SP {sum(r["forms"]["shortest"]["points"]["128"]["elapsed_seconds"] for r in load("holdout")["rows"]):.3f} s; ED {sum(r["forms"]["edge_disjoint"]["points"]["128"]["elapsed_seconds"] for r in load("holdout")["rows"]):.3f} s; native routad replay {native128:.3f} s. Native replay sker vid annan väggtid; gap/uppdateringar matchar exakt, tider är inte parad latensjämförelse.')
put('')
put('| Jobb | vägg s exkl kö | CPU s | max RSS KiB |')
put('|---|---:|---:|---:|')
for file in ['pilot.time','development.time','holdout.time','exact_pilot.time','flint_pilot.time','f1_decimal_diagnostic.time','f1.time','f1_successors.time','tree_voltage.time','remaining_initial_adapter.time','instance_tests.time','distance_quotient.time','port_conditioned.time','final_checks.time','f1_router_replay.time','f1_ram_correction.time']:
 text=(LANE/file).read_text();vals={}
 for label in ['User time (seconds)','System time (seconds)','Maximum resident set size (kbytes)','Elapsed (wall clock) time (h:mm:ss or m:ss)']:
  m=re.search(re.escape(label)+r':\s*([^\n]+)',text)
  if m:vals[label]=m.group(1)
 if len(vals)<4:continue
 ts=vals['Elapsed (wall clock) time (h:mm:ss or m:ss)'].split(':');wall=sum(float(t)*60**i for i,t in enumerate(reversed(ts)))
 cpu=float(vals['User time (seconds)'])+float(vals['System time (seconds)']);rss=int(vals['Maximum resident set size (kbytes)'])
 put(f'| {file} | {wall:.2f} | {cpu:.2f} | {rss} |')
put('')
put('GPU 0; trådar ≤2; ett beräkningsjobb åt gången, långkörningar via heavy_run. Första F1-körningen avbröts via kontrollerad egen PID för att rätta RAM-deklaration till uppmätt pilotpeak (~0,51 GB); avbrottslogg kvar. Preliminary Fraction-körningar med 120s-caps och decimal 10⁻¹⁸ tolkning bevaras som F1_DECIMAL_DIAGNOSTIC.json och exkluderas från slutsiffror; slutversionen har binär inputsemantik och facit i varje rad. Första tillagda testadapter packade en fyra-tuple som tre fält (9 adapterfel, inga inneslutningsfel); rättat och slutlig svit passerar. Vendor/cache/data ligger på angiven riktiga speldisk; manifest listar byte/hashes.')
put('')
put('## Nyhetsläge, reproducera och branch')
put('Kontrollerade originalsidor före grundbygget: [Kelner et al. 1301.6628](https://arxiv.org/abs/1301.6628), [Deweese et al. 1609.02957](https://arxiv.org/abs/1609.02957), [von Luxburg et al. 1003.1266](https://arxiv.org/abs/1003.1266). Cykel-/trädmetoder, degree-regimer, variational bounds och grafaggregation är kända. Fmpq-dokumentation: [python-flint](https://python-flint.readthedocs.io/en/latest/fmpq.html). Inget nytt öppet fältproblem eller ny generell grafalgoritm etableras. Det smala egna utfallet är den prövade routerregeln, dess F1-motexempel och en konkret återanvändbar potentialkonstruktion som ändrar vår engines mätta F1-slack.')
put('')
put('T1-bas 5717ace + fyra bevarade okommitterade filer blev egen bascommit **8bcdda9**. Egen branch `research/instance-slack-graph-20261001`, egen worktree. Slutlig kodcommit anges i CODE_COMMIT.txt. Källhashar oförändrade. Ingenting pushat, mergat eller registrerat i delad forskningsgraf. Alla prereg-filer bevaras före respektive operation; routerfreeze blev aldrig omsökt på holdout.')
put('')
put('Från denna lane, OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=2:')
put('```bash\n../../heavy_run.sh --ram-gb 0.13 -- python3 experiment.py development --count 20\npython3 analysis.py train\n../../heavy_run.sh --ram-gb 0.13 -- python3 experiment.py holdout --count 20\n../../heavy_run.sh --ram-gb 0.55 -- python3 f1_experiment.py run\n../../heavy_run.sh --ram-gb 0.55 -- python3 successors.py f1\n../../heavy_run.sh --ram-gb 0.55 -- python3 tree_voltage.py\n../../heavy_run.sh --ram-gb 0.13 -- python3 remaining.py\n../../heavy_run.sh --ram-gb 0.55 -- python3 distance_quotient.py\n../../heavy_run.sh --ram-gb 0.55 -- python3 distance_quotient.py --conditioned\n../../heavy_run.sh --ram-gb 0.55 -- python3 f1_router_replay.py\n../../heavy_run.sh --ram-gb 0.17 -- python3 final_checks.py\n```')
put('Observera: historiska remaining.py körde testadapterfelet; aktuell kod är rättad. Använd T9_DATA_DIR för nya råartefakter och installera python-flint lokalt där vid behov, så frysta receipt inte skrivs över. [SUMMARY.png]('+str(DATA/'SUMMARY.png')+') visar struktursambanden och den budgetmatchade F1-effekten.')
put('')
put('## Vad som föll → nästa avgörande försök')
put('Global hubbandel förklarar inte detta schema; G2 som allmän hubb-undvikanderegel faller, och G3:s positiva smågraph-holdout överförs inte till F1. Trädpotentialens nollfall på trädkanter betalar i stället med större chord energy. G4 visar starkt sidberoende och att hundrafaldig gapreduktion inte gav fler nära F1-tecken här. Kvotientens återstående within-class geometry och saknade native theta-workload är namngivna hinder, inte en slutstängd fråga.')
put('')
put('[NEXT.md](NEXT.md) ger tre körbara briefar: residualstyrd klassdelning för tecken, portberoende router med färska matchade grafer, och persistent operation under gemensam total kostnad. Den körda bytta operationen portkonditionerad kvotient gav fullbankens g=1,131 vid 128 lokala attempts och g=0,919 vid CG12; den separata supportändringen gav exakt tecken vid 0 attempts på den 4096-nodiga stjärnan.')
(LANE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
# Record compact executed attempts and successor mapping.
entries=[
 ('G1','degree/hub prediction','unchanged T1','edge count rho .539 holdout; hub share -.361','fixed-schedule coverage and terminal dependence','T9C_PORT_ROUTER'),
 ('G2','avoid maximum hub','8 greedy edge-disjoint paths','CG12 FULL3459 9.294->5.082; two hard cases unchanged','single alternative support and dual-side slack','T9B_THRESHOLD_QUOTIENT'),
 ('G3','single-threshold routing','frozen 3.245 native policy','holdout mean .216509->.205760; F1 loses to ED','size/port distribution transfer','T9C_PORT_ROUTER'),
 ('leaf','remove zero-current branches','leaf support + exact extension','star4096 gap2046.5->0 at 0; small F1 effect','F1 has large non-tree core','T9B_THRESHOLD_QUOTIENT'),
 ('warm_hub','central coordinate defect','two harmonic hub coordinates','small cycle gains; CG12 unchanged','defect spread across chord geometry','T9B_THRESHOLD_QUOTIENT'),
 ('tree_voltage','zero-flow tree potential','constant branches','F1 cycle gap worsens; CG12 unchanged','non-tree chord energy','T9B_THRESHOLD_QUOTIENT'),
 ('quotient','dual restricted energy','exact distance quotient','fullbank cycle128 975.906->1.740','boundary grouped with nuisance vertices','T9B_THRESHOLD_QUOTIENT'),
 ('conditioned','boundary representation','singleton ports + quotient','fullbank cycle128 1.131; CG12 .919; no new close signs','within-class geometry and theta margins','T9B_THRESHOLD_QUOTIENT')]
at=[{'id':i,'parent_source_ids':['T9_INSTANCE_SLACK_GRAPH','T1_CERT_RESISTANCE','T7_DECIDE_WITHOUT_SOLVING','PMAX_ROUTING'],'hypothesis':h,'changed_operation':op,'executed_outcome':v,'binding_requirement':b,'next_brief':n,'evidence':str(DATA)} for i,h,op,v,b,n in entries]
(LANE/'ATTEMPTS.json').write_text(json.dumps(at,indent=2)+'\n');(LANE/'SEED_EXPANSIONS.json').write_text(json.dumps({'expansions':at,'next_locator':str(LANE/'NEXT.md')},indent=2)+'\n')
