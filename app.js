'use strict';
const films = [
  {title:'Eight basic PDEs',duration:120,figure:'Fig. 2',description:'Follow eight classical systems from observed fields through symbolic search to recovered equations and predictions under held-out conditions.',file:'eight_basic_PDEs'},
  {title:'Five open-form PDEs',duration:112,figure:'Fig. 3',description:'Explore how recursive algebraic and differential operations reveal nested nonlinear structures, from gravity flow to graph-curvature dynamics.',file:'five_open_form_PDEs'},
  {title:'Three-dimensional Euler equations',duration:62,figure:'Fig. 4a–e',description:'Trace shared symbolic computation through the recovery of five coupled conservation laws and the prediction of three-dimensional compressible flow.',file:'three_dimensional_Euler'},
  {title:'Navier–Stokes and Gray–Scott',duration:120,figure:'Fig. 4',description:'Examine coupled flow and reaction–diffusion dynamics as observations become noisy or spatially sparse, linking equation selection to predictive tests.',file:'Navier_Stokes_Gray_Scott'},
  {title:'Standing-wave tube',duration:77,figure:'Fig. 5a–c',description:'See sparse pressure and particle-velocity measurements reveal coupled acoustic equations, an inferred sound speed and a consistent energy balance.',file:'standing_wave_tube'},
  {title:'MeshRIR acoustic field',duration:80,figure:'Fig. 5d',description:'Recover directional wave propagation from measured room acoustics and compare the inferred sound speed with the sound speed calculated from the dataset’s recorded temperature.',file:'MeshRIR_3D_acoustic_field',source:'Koyama et al. · 2021',doi:'10.1109/WASPAA52581.2021.9632672'},
  {title:'Cylinder wake',duration:72,figure:'Fig. 5e',description:'Follow measured velocity fields into a conservative vorticity-transport equation, then compare its integrated predictions with withheld experimental observations.',file:'cylinder_wake',source:'Renn et al. · 2026',doi:'10.1038/s41597-026-07368-z'},
  {title:'Gel electrophoresis',duration:72,figure:'Fig. 5f',description:'Track image-based concentration patterns through equation discovery, revealing directional diffusion and drift before evaluating the resulting future-field predictions.',file:'gel_electrophoresis',source:'Both et al. · 2021',doi:'10.1016/j.jcp.2020.109985'},
  {title:'Near-breaking surface waves',duration:72,figure:'Fig. 5g',description:'Follow measured surface elevation into a compact three-term equation that combines propagation, dispersion and a differentiated nonlinear contribution.',file:'near_breaking_surface_waves',source:'Xu et al. · 2025',doi:'10.1038/s41467-025-65114-2'},
  {title:'Active nematic dynamics',duration:72,figure:'Fig. 5h–i',description:'Connect measured flow and orientation to two coupled equations describing how advection, rotation and alignment shape active-matter dynamics.',file:'active_nematic_dynamics',source:'Golden et al. · 2023',doi:'10.1126/sciadv.abq6120'}
];
const chineseFilms = [
  ['八个基本偏微分方程','从八类经典系统的观测场出发，展示符号搜索、方程恢复及未参与建模条件下的预测检验。'],
  ['五个开放形式偏微分方程','通过递归组合代数运算与微分运算，揭示从重力流到曲率驱动演化中的嵌套非线性结构。'],
  ['三维欧拉方程','展示共享符号计算如何恢复五条耦合守恒方程，并预测三维可压缩流动。'],
  ['纳维–斯托克斯与 Gray–Scott 方程','在含噪声或空间稀疏的观测下，展示流动与反应扩散方程的选择及预测检验。'],
  ['驻波管实验','从稀疏声压与质点振速测量中恢复耦合声学方程，推导声速及相应的能量守恒关系。'],
  ['MeshRIR 三维声场','从室内实测声场恢复各方向的波动传播，并将反推声速与数据集实测温度折算的声速比较。'],
  ['圆柱尾流','从实测速度场恢复守恒形式的涡量输运方程，并比较方程积分结果与留出的实验观测。'],
  ['凝胶电泳','从图像中的浓度变化识别方向性扩散和漂移，并检验方程对后续浓度场的预测。'],
  ['近破碎表面波','从实测表面高度恢复包含传播、色散及非线性微分项的三项方程。'],
  ['活性向列相动力学','从实测流场与取向场恢复两条耦合方程，描述平流、旋转和取向对齐对活性物质演化的作用。']
];
const translations = {
  en: {
    skip:'Skip to player',supplements:'Supplementary films',collection:'The collection',study:'The study',
    eyebrow:'A visual companion to SPADE',headline:'From fields to equations',
    introFirst:'Watch physical laws emerge from data.',introSecond:'Ten films on symbolic discovery and prediction.',
    count:'10 films',totalTime:'14 min 19 s',download:'Download film',collectionHeading:'In this collection',
    paperFirst:'Discovering partial differential equations',paperSecond:'through shared symbolic computation',
    studyDescription:'SPADE reuses shared symbolic computations to search for governing equations. These films follow the journey from observed fields through equation discovery to validation, across numerical systems and six physical experiments.',
    backTop:'Back to top ↑',numerical:'Numerical systems',physical:'Physical experiments',movie:'Supplementary Movie',
    select:'Select Movie',play:'Play',data:'Data:',home:'SPADE home',navigation:'Page navigation',
    screening:'Supplementary movie collection',choose:'Choose a film',previous:'Previous film',next:'Next film',
    fallback:'Your browser does not support this video. Use the download link below.',
    loadError:'The film could not be loaded. You can download it using the link below.',playPrompt:'Press the video play control to begin.',
    meta:'Ten supplementary films following SPADE from observed fields to governing equations and predictive tests.'
  },
  zh: {
    skip:'跳转到播放器',supplements:'补充视频',collection:'视频目录',study:'关于研究',
    eyebrow:'SPADE 研究视频',headline:'从观测场到方程',
    introFirst:'看物理规律如何从数据中浮现。',introSecond:'十部视频，呈现方程发现与预测检验。',
    count:'10 部视频',totalTime:'14 分 19 秒',download:'下载视频',collectionHeading:'视频目录',
    paperFirst:'通过共享符号计算',paperSecond:'发现偏微分方程',
    studyDescription:'SPADE 复用不同候选表达式中的相同计算，寻找支配系统演化的方程。这十部视频覆盖数值系统与六类真实实验，展示从观测场、方程发现到预测检验的全过程。',
    backTop:'返回顶部 ↑',numerical:'数值系统',physical:'真实实验',movie:'补充视频',
    select:'选择视频',play:'播放',data:'数据：',home:'SPADE 首页',navigation:'页面导航',
    screening:'补充视频集',choose:'选择视频',previous:'上一部视频',next:'下一部视频',
    fallback:'当前浏览器不支持视频播放，请使用下方链接下载。',
    loadError:'视频加载失败，可使用下方链接下载。',playPrompt:'请点击视频播放按钮开始播放。',
    meta:'十部 SPADE 补充视频，展示从观测场、方程发现到预测检验的全过程。'
  }
};
let language='en';
try { if(localStorage.getItem('spade-film-language')==='zh') language='zh'; } catch {}
const t = key => translations[language][key];
const filmText = index => language==='zh' ? chineseFilms[index] : [films[index].title,films[index].description];
const byId = id => document.getElementById(id);
const number = index => String(index+1).padStart(2,'0');
const duration = seconds => `${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
const player = byId('player');
const playButton = byId('play');
const message = byId('playback-message');
let selected = -1;

films.forEach((film,index) => {
  if(index===0 || index===4){
    const heading=document.createElement('h3');
    heading.className='group-label';
    heading.dataset.group=index===0?'numerical':'physical';
    byId('film-list').append(heading);
  }
  const button=document.createElement('button');
  button.className='film-choice';
  button.dataset.index=index;
  button.setAttribute('aria-controls','player');
  button.innerHTML=`<span class="film-number">${number(index)}</span><img src="assets/posters/movie-${number(index)}.jpg" alt="" width="128" height="72" loading="lazy"><span class="film-name"><span class="film-title-text"></span><small>${duration(film.duration)}</small></span><span class="film-arrow" aria-hidden="true">↗</span>`;
  button.addEventListener('click',() => selectFilm(index,true));
  byId('film-list').append(button);
});

function renderFilmText(){
  if(selected<0) return;
  const [title,description]=filmText(selected);
  const film=films[selected];
  player.setAttribute('aria-label',title);
  playButton.setAttribute('aria-label',`${t('play')} ${title}`);
  byId('screen-label').textContent=`${t('movie')} ${number(selected)}`;
  byId('screen-type').textContent=t(selected<4?'numerical':'physical');
  byId('film-title').textContent=title;
  byId('film-description').textContent=description;
  byId('figure').textContent=language==='zh'?film.figure.replace('Fig.','图'):film.figure;
  if(film.doi) byId('data-source').textContent=`${t('data')} ${film.source} ↗`;
}

function setLanguage(value){
  language=value;
  document.documentElement.lang=language==='zh'?'zh-CN':'en';
  document.title=`SPADE — ${t('headline')}`;
  document.querySelector('meta[name="description"]').content=t('meta');
  document.querySelectorAll('[data-i18n]').forEach(element => {element.textContent=t(element.dataset.i18n);});
  document.querySelectorAll('[data-i18n-label]').forEach(element => {element.setAttribute('aria-label',t(element.dataset.i18nLabel));});
  document.querySelectorAll('.group-label').forEach(heading => {
    heading.textContent=`${heading.dataset.group==='numerical'?'01–04':'05–10'} / ${t(heading.dataset.group)}`;
  });
  document.querySelectorAll('.film-choice').forEach((button,index) => {
    const [title]=filmText(index);
    button.querySelector('.film-title-text').textContent=title;
    button.setAttribute('aria-label',`${t('select')} ${number(index)}: ${title}`);
  });
  const toggle=byId('language-toggle');
  toggle.textContent=language==='zh'?'English':'中文';
  toggle.lang=language==='zh'?'en':'zh-CN';
  toggle.setAttribute('aria-label',language==='zh'?'Switch to English':'切换为中文');
  renderFilmText();
  if(message.dataset.message) message.textContent=t(message.dataset.message);
  try { localStorage.setItem('spade-film-language',language); } catch {}
}

function showMessage(key){
  message.dataset.message=key;
  message.textContent=t(key);
  message.hidden=false;
}

function play(){
  message.hidden=true;
  player.play().catch(() => showMessage(player.error?'loadError':'playPrompt'));
}

function selectFilm(index,autoplay=false){
  if(!Number.isInteger(index) || index<0 || index>=films.length) return;
  if(index===selected){ if(autoplay) play(); return; }
  const film=films[index];
  player.pause();
  player.controls=false;
  selected=index;
  player.poster=`assets/posters/movie-${number(index)}.jpg`;
  player.src=`assets/videos/movie-${number(index)}.mp4`;
  playButton.hidden=false;
  renderFilmText();
  byId('film-duration').textContent=duration(film.duration);
  byId('position').textContent=`${number(index)} / 10`;
  byId('previous').disabled=index===0;
  byId('next').disabled=index===films.length-1;
  byId('download').href=player.getAttribute('src');
  byId('download').download=`Supplementary_Movie_${number(index)}_${film.file}.mp4`;
  const source=byId('data-source');
  source.hidden=!film.doi;
  if(film.doi) source.href=`https://doi.org/${film.doi}`;
  else source.removeAttribute('href');
  document.querySelectorAll('.film-choice').forEach((button,i) => button.setAttribute('aria-current',String(i===index)));
  message.hidden=true;
  history.replaceState(null,'',`#movie-${number(index)}`);
  if(autoplay){
    play();
    if(matchMedia('(max-width: 820px)').matches) byId('screening').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
  }
}

playButton.addEventListener('click',play);
player.addEventListener('play',() => {player.controls=true;playButton.hidden=true;message.hidden=true;});
player.addEventListener('error',() => showMessage('loadError'));
byId('language-toggle').addEventListener('click',() => setLanguage(language==='en'?'zh':'en'));
byId('previous').addEventListener('click',() => selectFilm(selected-1,true));
byId('next').addEventListener('click',() => selectFilm(selected+1,true));
function openHash(){
  const match=location.hash.match(/^#movie-(\d{2})$/);
  const index=match?Number(match[1])-1:0;
  selectFilm(index>=0 && index<films.length?index:0);
}
window.addEventListener('hashchange',() => { if(location.hash.startsWith('#movie-')) openHash(); });
setLanguage(language);
openHash();
