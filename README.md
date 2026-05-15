## install 
git clone https://github.com/3MPER0RR/RamRouteAi/

cd RamRouteAi/project

touch .env

nano .env

insert in .env 

LLM_API_URL=https://api.groq.com/openai/v1/chat/completions

LLM_API_KEY=insert_apikey

LLM_MODEL=llama-3.3-70b-versatile

python3.11 -m venv venv

source venv/bin/activate

pip install -r requirements

# Requirements

- Python 3.11+

- Core dependencies:

numpy,httpx,python-dotenv

uvloop,orjson,scikit-learn,rich,psutil

RamRouteAi
A RAM-first routing layer for LLM APIs — reducing latency and API call volume through semantic caching.


Every call to a remote LLM API introduces latency — typically between 300ms and 2 seconds per response — along with a per-token cost. In many real-world applications, a significant portion of incoming queries are semantically redundant: different phrasings of the same intent, repeated questions across sessions, or variations on a narrow domain of inputs. Each of these still pays the full cost of a remote round-trip.


RamRouteAi sits between your application and the LLM API. Incoming queries are embedded and compared against a semantic cache held entirely in RAM. If a query is sufficiently similar to a previously seen one — above a configurable similarity threshold — the cached response is returned locally, with sub-millisecond latency. Only genuinely novel queries are forwarded to the external API.

The routing decision is made by a lightweight similarity classifier trained on past query-response pairs. The system supports Groq and any OpenAI-compatible endpoint as its external fallback.

Performance
(Benchmark results to be added — target metrics: cache hit rate on representative workloads, mean latency with and without cache hit, RAM usage at varying cache sizes.)


Status
Experimental prototype under active development.

Active development — modular components are being incrementally extended.
