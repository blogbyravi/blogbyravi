const articles = [
  {
    slug: 'rag-chatbots-that-earn-trust',
    title: 'RAG chatbots that earn trust',
    category: 'Generative AI',
    date: 'AUG 02, 2026',
    readTime: '6 MIN READ',
    description: 'A useful RAG system is less about a clever prompt and more about the quality of every step around it.',
    image: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&w=640&q=80',
    alt: 'Abstract generative AI visualization',
    color: 'lime',
    dek: 'A practical look at retrieval, grounding, and evaluation for AI assistants that need to be useful beyond a demo.'
  },
  {
    slug: 'from-notebook-to-production',
    title: 'From notebook to production: the handoff that matters',
    category: 'Machine Learning',
    date: 'SEP 18, 2026',
    readTime: '7 MIN READ',
    description: 'Reliable machine learning starts when reproducibility and monitoring become part of the model design.',
    image: 'https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=640&q=80',
    alt: 'A close view of a computer circuit board',
    color: 'coral',
    dek: 'The practical habits that help a promising experiment become a repeatable, observable service.'
  },
  {
    slug: 'machine-learning-model-deployment-aws-ec2',
    markdownFile: 'End_to_End_Deployement_For_ML.md',
    title: 'Machine learning model deployment: from notebook to AWS EC2',
    category: 'Machine Learning',
    date: 'OCT 05, 2026',
    readTime: '18 MIN READ',
    description: 'A practical end-to-end guide to taking a model from notebook experiments to an AWS EC2 service.',
    image: 'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=640&q=80',
    alt: 'Rows of server racks in a data center',
    color: 'green',
    dek: 'Move a machine learning model from notebook to a usable service with reusable Python code, Streamlit, FastAPI, and AWS EC2.'
  },
  {
    slug: 'house-price-prediction-using-machine-learning',
    markdownFile: 'Use_Case_House_Price_Prediction.md',
    title: 'House price prediction using machine learning',
    category: 'Machine Learning',
    date: 'JULY 02, 2026',
    readTime: '20 MIN READ',
    description: 'An end-to-end California housing regression project, from preprocessing and cross-validation to model evaluation.',
    image: 'https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=640&q=80',
    alt: 'A residential home representing the house-price prediction project',
    color: 'green',
    dek: 'A complete use-case report covering the California housing dataset, model selection, HistGradientBoosting, evaluation, and practical limitations.'
  },
  {
    slug: 'analytics-that-change-a-decision',
    title: 'Analytics that change a decision',
    category: 'Analytics',
    date: 'SEP 04, 2026',
    readTime: '5 MIN READ',
    description: 'A dashboard is not the outcome. The outcome is a clearer decision, made with less friction.',
    image: 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=640&q=80',
    alt: 'Analytics charts on a laptop screen',
    color: 'blue',
    dek: 'Start with the decision and work backward: a more useful way to scope metrics, analysis, and dashboards.'
  },
  {
    slug: 'cloud-ml-with-sagemaker-s3-and-boto3',
    title: 'A grounded cloud ML workflow with SageMaker and S3',
    category: 'Cloud & ML',
    date: 'JUNE 21, 2026',
    readTime: '8 MIN READ',
    description: 'A clear path from versioned data to a repeatable training run, using familiar AWS building blocks.',
    image: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=640&q=80',
    alt: 'Earth at night with network connections',
    color: 'green',
    dek: 'How S3, Boto3, and SageMaker can fit together into a workflow that is easier to reproduce and operate.'
  }
];

const grid = document.querySelector('#article-grid');
const filters = document.querySelector('#filters');
const searchInput = document.querySelector('#article-search');
const emptyState = document.querySelector('#empty-state');
const homeView = document.querySelector('body > main:not(#article-view)');
const articleView = document.querySelector('#article-view');
const categories = ['All notes', ...new Set(articles.map((article) => article.category))];
let activeCategory = 'All notes';
let searchTerm = '';

function renderFilters() {
  filters.innerHTML = categories.map((category) => `
    <button class="filter-button" type="button" data-category="${category}" aria-pressed="${category === activeCategory}">${category}</button>
  `).join('');
}

function renderArticles() {
  const matching = articles.filter((article) => {
    const matchesCategory = activeCategory === 'All notes' || article.category === activeCategory;
    const searchContent = `${article.title} ${article.category} ${article.description}`.toLowerCase();
    return matchesCategory && searchContent.includes(searchTerm.toLowerCase());
  });

  grid.innerHTML = matching.map((article) => `
    <article class="article-card">
      <a class="card-image" href="?article=${encodeURIComponent(article.slug)}" tabindex="-1" aria-hidden="true">
        <img src="${article.image}" alt="${article.alt}" loading="lazy">
      </a>
      <div class="card-content">
        <p class="card-meta"><span class="card-category">${article.category.toUpperCase()}</span><span>·</span><time>${article.date}</time></p>
        <h3 class="card-title"><a href="?article=${encodeURIComponent(article.slug)}">${article.title}</a></h3>
        <p class="card-description">${article.description}</p>
        <a class="card-read" href="?article=${encodeURIComponent(article.slug)}">Read note <span aria-hidden="true">↗</span><span class="sr-only">: ${article.title}</span></a>
      </div>
    </article>
  `).join('');
  emptyState.hidden = matching.length > 0;
}

function showHome() {
  homeView.hidden = false;
  articleView.hidden = true;
  delete articleView.dataset.articleSlug;
  document.title = 'BlogByRavi | Data Science & AI';
  window.scrollTo(0, 0);
}

function renderMarkdown(markdown) {
  if (window.marked) {
    window.marked.setOptions({ gfm: true, breaks: false });
    const parsed = new DOMParser().parseFromString(window.marked.parse(markdown), 'text/html');
    const usedIds = new Set();
    parsed.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach((heading) => {
      const baseId = heading.id || heading.textContent.trim().toLowerCase()
        .replace(/[^\p{L}\p{N}\s_-]/gu, '')
        .replace(/\s/g, '-');
      let id = baseId;
      let suffix = 1;
      while (usedIds.has(id)) id = `${baseId}-${suffix++}`;
      heading.id = id;
      usedIds.add(id);
    });
    return parsed.body.innerHTML;
  }
  const escaped = markdown.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
  return `<pre>${escaped}</pre>`;
}

function scrollToArticleFragment() {
  const fragment = decodeURIComponent(location.hash.slice(1));
  if (!fragment) return false;
  const target = document.getElementById(fragment);
  if (!target) return false;
  target.scrollIntoView({ behavior: 'instant', block: 'start' });
  return true;
}

async function showArticle(slug, shouldPush = false) {
  const article = articles.find((item) => item.slug === slug);
  if (!article) {
    showHome();
    return;
  }
  if (shouldPush) history.pushState({ article: slug }, '', `?article=${encodeURIComponent(slug)}`);
  articleView.dataset.articleSlug = slug;
  homeView.hidden = true;
  articleView.hidden = false;
  articleView.innerHTML = `
    <a class="back-link" href="./"><span aria-hidden="true">←</span> ALL NOTES</a>
    <header class="article-header">
      <p class="eyebrow"><span class="status-dot"></span>${article.category.toUpperCase()} <span>·</span> ${article.date}</p>
      <h1>${article.title}</h1>
      <p class="article-dek">${article.dek}</p>
      <p class="article-byline">${article.readTime} &nbsp;·&nbsp; BLOGBYRAVI</p>
    </header>
    <img class="article-cover" src="${article.image.replace('w=640', 'w=1400')}" alt="${article.alt}">
    <div class="article-body" aria-live="polite"><p>Loading article…</p></div>
  `;
  document.title = `${article.title} | BlogByRavi`;
  window.scrollTo(0, 0);

  try {
    const response = await fetch(`articles/${article.markdownFile || `${article.slug}.md`}`);
    if (!response.ok) throw new Error(`Article request failed (${response.status})`);
    const markdown = await response.text();
    const body = articleView.querySelector('.article-body');
    if (body) body.innerHTML = renderMarkdown(markdown);
    scrollToArticleFragment();
  } catch (error) {
    const body = articleView.querySelector('.article-body');
    if (body) body.innerHTML = '<p>This note could not be loaded. Please open the site through a local web server so the Markdown file can be fetched.</p>';
    console.error(error);
  }
}

filters.addEventListener('click', (event) => {
  const button = event.target.closest('[data-category]');
  if (!button) return;
  activeCategory = button.dataset.category;
  renderFilters();
  renderArticles();
});

searchInput.addEventListener('input', () => {
  searchTerm = searchInput.value.trim();
  renderArticles();
});

document.addEventListener('click', (event) => {
  const link = event.target.closest('.article-body a[href^="#"]');
  if (!link) return;
  const target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
  if (!target) return;
  event.preventDefault();
  if (location.hash !== link.hash) {
    history.pushState(history.state, '', `${location.pathname}${location.search}${link.hash}`);
  }
  scrollToArticleFragment();
});

document.addEventListener('click', (event) => {
  const link = event.target.closest('a[href^="?article="]');
  if (!link) return;
  event.preventDefault();
  const slug = new URL(link.href).searchParams.get('article');
  showArticle(slug, true);
});

document.querySelector('.back-link')?.addEventListener('click', (event) => event.preventDefault());

document.addEventListener('click', (event) => {
  if (!event.target.closest('.back-link')) return;
  event.preventDefault();
  history.pushState({}, '', location.pathname);
  showHome();
});

window.addEventListener('popstate', () => {
  const slug = new URLSearchParams(location.search).get('article');
  if (!slug) {
    showHome();
    return;
  }
  if (articleView.dataset.articleSlug === slug && !articleView.hidden) {
    if (!scrollToArticleFragment()) window.scrollTo(0, 0);
    return;
  }
  showArticle(slug);
});

document.querySelector('.menu-toggle').addEventListener('click', (event) => {
  const button = event.currentTarget;
  const open = button.getAttribute('aria-expanded') !== 'true';
  button.setAttribute('aria-expanded', String(open));
  document.querySelector('.main-nav').classList.toggle('is-open', open);
});

document.querySelectorAll('.main-nav a[href^="#"]').forEach((link) => {
  link.addEventListener('click', (event) => {
    document.querySelector('.main-nav').classList.remove('is-open');
    document.querySelector('.menu-toggle').setAttribute('aria-expanded', 'false');

    if (!articleView.hidden) {
      event.preventDefault();
      history.pushState({}, '', `${location.pathname}${link.hash}`);
      showHome();
      document.querySelector(link.hash)?.scrollIntoView({ behavior: 'smooth' });
    }
  });
});

document.addEventListener('keydown', (event) => {
  if (event.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
    event.preventDefault();
    searchInput.focus();
  }
});

document.querySelector('#year').textContent = new Date().getFullYear();
renderFilters();
renderArticles();
const initialSlug = new URLSearchParams(location.search).get('article');
if (initialSlug) showArticle(initialSlug);
