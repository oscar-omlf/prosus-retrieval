'use strict';

const $ = id => document.getElementById(id);
const state = {meta: null, busy: false, request: null, response: null};
const currency = new Intl.NumberFormat('pt-BR', {style: 'currency', currency: 'BRL'});
const number = new Intl.NumberFormat('en-US');

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.classList.add('icon'); svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#' + name + '-icon'); svg.append(use);
  return svg;
}

function json(id, value) { $(id).textContent = JSON.stringify(value, null, 2); }
function modeName(mode) { return state.meta?.modes.find(value => value.id === mode)?.name || mode; }
function ms(value) { return typeof value === 'number' && Number.isFinite(value) ? number.format(Math.round(value)) + ' ms' : 'Not reported'; }
function announce(text) { $('live-status').textContent = text; }

function empty(title, description) {
  const box = node('div', 'empty-state');
  const art = node('div', 'empty-art'); art.append(icon('plate')); box.append(art);
  box.append(node('h3', '', title), node('p', '', description));
  $('results').replaceChildren(box);
}

function imageFallback(media) {
  const fallback = node('div', 'fallback');
  fallback.append(icon('plate'), node('span', '', 'No catalog image'));
  media.append(fallback);
}

function renderProduct(product) {
  const card = node('article', 'product-card' + (product.rank === 1 ? ' first' : ''));
  const media = node('div', 'product-media');
  const url = product.image_url;
  if (url && url.startsWith('https://static.ifood-static.com.br/image/upload/t_low/pratos/')) {
    const image = node('img'); image.src = url; image.alt = product.name;
    image.loading = 'lazy'; image.decoding = 'async'; image.referrerPolicy = 'no-referrer';
    image.addEventListener('error', () => { image.remove(); imageFallback(media); }, {once: true});
    media.append(image);
  } else imageFallback(media);
  const rank = node('span', 'product-rank', String(product.rank)); rank.setAttribute('aria-label', 'Rank ' + product.rank);
  media.append(rank); card.append(media);
  const body = node('div', 'product-body');
  body.append(node('div', 'product-category', product.category || 'Category not specified'),
    node('h3', '', product.name), node('p', 'product-description', product.description || 'No description provided.'));
  const bottom = node('div', 'product-bottom');
  const priced = typeof product.price === 'number' && Number.isFinite(product.price) && product.price > 0;
  bottom.append(node('span', 'price' + (priced ? '' : ' unknown'), priced ? currency.format(product.price) : 'Price not specified'));
  const score = typeof product.score === 'number' ? product.score : null;
  const label = product.score_kind === 'contextual-ordinal' ? 'Order score' : 'Rank score';
  const badge = node('span', 'score', label + ' ' + (score === null ? '—' : Number(score.toFixed(4))));
  badge.title = (product.score_kind || 'Ranking score') + ' · not a probability'; bottom.append(badge); body.append(bottom);
  const details = node('details', 'product-details'); details.append(node('summary', '', 'Product details & evidence'));
  details.append(node('p', 'full-description', product.description || 'No description provided.'));
  const data = {item_id: product.item_id, merchant_id: product.merchant_id, score: product.score,
    score_kind: product.score_kind, explanation: product.explanation};
  details.append(node('pre', '', JSON.stringify(data, null, 2))); body.append(details); card.append(body);
  return card;
}

function setBusy(busy) {
  state.busy = busy;
  $('search-button').disabled = busy;
  $('search-button').querySelector('span').textContent = busy ? 'Searching…' : 'Search';
  $('mode').disabled = busy || !state.meta;
  document.querySelectorAll('.example').forEach(button => { button.disabled = busy; });
  $('results-section').setAttribute('aria-busy', String(busy));
}

function updateMode() {
  const mode = state.meta?.modes.find(value => value.id === $('mode').value);
  if (!mode) return;
  let description = mode.description;
  if (mode.id === 'contextual' && !state.meta.remote_api_enabled) description += ' API disabled: validated cached requests only.';
  $('mode-description').textContent = description;
}

async function refreshMetadata(initial = false) {
  const response = await fetch('/api/meta', {credentials: 'omit'});
  if (!response.ok) throw new Error('The local search service is unavailable. Check the server and reload.');
  const meta = await response.json(); state.meta = meta;
  $('connection').classList.remove('offline'); $('connection').lastChild.textContent = 'Local service ready';
  $('catalog-count').textContent = number.format(meta.catalog_count);
  $('budget-value').textContent = '$' + meta.budget.accounted_usd.toFixed(4) + ' / $' + meta.budget.ceiling_usd.toFixed(2);
  $('budget-meter').max = meta.budget.ceiling_usd; $('budget-meter').value = meta.budget.accounted_usd;
  $('api-note').textContent = meta.remote_api_enabled
    ? 'Contextual mode can make budgeted model calls. Local modes do not call the model API.'
    : 'Model API disabled. Local modes and validated cached contextual requests remain available.';
  if (initial) {
    const params = new URLSearchParams(location.search);
    $('mode').replaceChildren(...meta.modes.map(mode => {
      const option = node('option', '', mode.name); option.value = mode.id; return option;
    }));
    $('mode').value = meta.modes.some(mode => mode.id === params.get('mode')) ? params.get('mode') : meta.mode;
    $('mode').disabled = false;
    $('query').value = params.get('q') || meta.examples[0] || '';
    const examples = [node('span', 'example-label', 'TRY A QUERY')];
    [meta.examples[0], meta.examples[1], meta.examples[3]].filter(Boolean).forEach(query => {
      const button = node('button', 'example', query); button.type = 'button'; button.lang = 'pt-BR';
      button.append(node('span', 'arrow', '↗'));
      button.addEventListener('click', () => { $('query').value = query; $('search-form').requestSubmit(); });
      examples.push(button);
    });
    $('examples').replaceChildren(...examples); updateMode();
    // Opening the page never triggers a paid search; submit explicitly.
  }
}

async function search(event) {
  event?.preventDefault();
  if (state.busy || !state.meta) return;
  const query = $('query').value.trim();
  if (!query) { $('query').focus(); $('query').reportValidity(); return; }
  const request = {q: query, mode: $('mode').value, k: 5}; state.request = request; state.response = null;
  json('request-json', request); $('response-json').textContent = 'Waiting for the server…';
  $('interpretation-json').textContent = 'Waiting for the server…';
  document.querySelector('[data-copy="request-json"]').disabled = false;
  document.querySelector('[data-copy="response-json"]').disabled = true;
  for (const id of ['stat-server','stat-roundtrip','stat-cache','stat-candidates','stat-api']) $(id).textContent = '—';
  $('stat-mode').textContent = modeName(request.mode); $('release-id').textContent = 'Waiting for the server…';
  $('error').hidden = true; setBusy(true); announce('Searching for ' + query);
  $('results-title').textContent = 'Finding your matches';
  $('result-summary').textContent = 'Searching the real catalog…';
  const skeletons = Array.from({length: 4}, () => { const item = node('div', 'skeleton'); item.setAttribute('aria-hidden', 'true'); return item; });
  $('results').replaceChildren(...skeletons);
  const params = new URLSearchParams({q: query, mode: request.mode, k: '5'});
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 120000);
  const started = performance.now();
  try {
    const response = await fetch('/search?' + params, {signal: controller.signal, credentials: 'omit'});
    const data = await response.json(); state.response = data;
    $('stat-roundtrip').textContent = ms(performance.now() - started);
    json('response-json', data); document.querySelector('[data-copy="response-json"]').disabled = false;
    if (!response.ok) throw new Error(data.error || 'The server could not complete this search.');
    $('stat-mode').textContent = modeName(data.mode);
    $('stat-server').textContent = ms(data.http_server_ms ?? data.timing_ms?.total);
    $('stat-cache').textContent = data.reranking ? (data.reranking.cache_hit ? 'Hit' : 'Miss') : 'Not reported';
    $('stat-candidates').textContent = data.reranking?.candidate_count ?? 'Not reported';
    $('stat-api').textContent = data.remote_api_enabled ? 'Enabled' : 'Disabled';
    $('release-id').textContent = data.release_id || 'Not reported';
    json('interpretation-json', data.interpretation || {note: 'This mode does not return a structured interpretation.'});
    $('results-title').textContent = 'Results · ' + modeName(data.mode);
    $('result-summary').textContent = data.results.length + ' matches · ' + ms(data.http_server_ms ?? data.timing_ms?.total);
    if (data.results.length) $('results').replaceChildren(...data.results.map(renderProduct));
    else empty('No compatible products found', 'Try a broader request or review the query interpretation.');
    announce(data.results.length + ' results returned for ' + query);
    history.replaceState(null, '', '/?' + new URLSearchParams({q: query, mode: request.mode}));
  } catch (error) {
    let message = error.name === 'AbortError'
      ? 'The request timed out. The server may still be processing it; check API status before retrying.'
      : error.message || 'Could not reach the local server. Check that it is running.';
    if (request.mode === 'constraint-hybrid' && message.includes('offline_cache_miss')) {
      message = 'Constraint-aware mode handles simple queries offline. This request needs a cached interpretation. Choose Contextual for uncached complex requests; the original server error is in Response JSON.';
    }
    $('error-message').textContent = message; $('error').hidden = false;
    if (!state.response) { json('response-json', {error: message}); document.querySelector('[data-copy="response-json"]').disabled = false; }
    $('interpretation-json').textContent = 'No successful interpretation returned.';
    $('results-title').textContent = 'Search not completed'; $('result-summary').textContent = 'Review the message above';
    empty('Let’s try that again', 'Adjust your query or search mode. The server error is available above.');
    announce('Search failed. ' + message);
  } finally {
    clearTimeout(timeout); setBusy(false);
    refreshMetadata().catch(() => { $('connection').classList.add('offline'); $('connection').lastChild.textContent = 'Service unavailable'; });
  }
}

$('search-form').addEventListener('submit', search);
$('mode').addEventListener('change', updateMode);
$('retry').addEventListener('click', () => $('search-form').requestSubmit());
document.addEventListener('keydown', event => {
  const editing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
  if (event.key === '/' && !editing && !event.ctrlKey && !event.metaKey && !event.altKey) {
    event.preventDefault(); $('query').focus(); $('query').select();
  }
});
let toastTimer;
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  let message;
  try { await navigator.clipboard.writeText($(button.dataset.copy).textContent); message = 'Copied to clipboard'; }
  catch { message = 'Clipboard unavailable. Select and copy the JSON directly.'; }
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3000);
}));

refreshMetadata(true).catch(error => {
  $('connection').classList.add('offline'); $('connection').lastChild.textContent = 'Service unavailable';
  $('error-title').textContent = 'Could not connect'; $('error-message').textContent = error.message;
  $('error').hidden = false; $('search-button').disabled = true; $('retry').textContent = 'Reload page';
  $('retry').addEventListener('click', () => location.reload(), {once: true});
});
