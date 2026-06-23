import os
import pandas as pd
import streamlit as st

from library_engine import LibrarySearchEngine, HISTORY_CSV

st.set_page_config(page_title="Library Search", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading indices and models ...")
def get_engine():
    return LibrarySearchEngine.load(with_rerank=True)


if "history" not in st.session_state:
    st.session_state.history = []   # list of {"role","content"}

engine = get_engine()


with st.sidebar:
    st.header("Filters")
    f_author = st.text_input("Author contains")
    f_subjects = st.text_input("Subject contains")
    types = sorted(t for t in engine.corpus["ItemType"].unique() if t)
    f_types = st.multiselect("Item type", types)
    yr_lo = int(engine.corpus["Year"].replace(0, pd.NA).min() or 1900)
    yr_hi = int(engine.corpus["Year"].max() or 2025)
    f_years = st.slider("Year range", yr_lo, yr_hi, (yr_lo, yr_hi))

    st.header("Search options")
    use_rerank = st.toggle("Cross-encoder reranking", value=True)
    use_mq = st.toggle("Multi-query expansion (LLM)", value=False,
                       disabled=engine.llm is None,
                       help="Requires a local GGUF model via LLM_GGUF")
    top_k = st.slider("Results", 3, 20, 8)

    st.divider()
    st.caption("LLM: " + ("loaded ✅" if engine.llm else "not set (list-only answers)"))


def build_filters():
    f = {}
    if f_author: f["author"] = f_author
    if f_subjects: f["subjects"] = f_subjects
    if f_types: f["item_type"] = f_types
    if f_years[0] > yr_lo: f["year_min"] = f_years[0]
    if f_years[1] < yr_hi: f["year_max"] = f_years[1]
    return f



tab_search, tab_avail, tab_analytics = st.tabs(["🔎 Search", "📍 Availability", "📊 Analytics"])

with tab_search:
    st.title("📚 Library Search")

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    query = st.chat_input("Ask about the catalogue ...")
    if query:
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching ..."):
                results, meta = engine.search(
                    query, k=top_k, filters=build_filters(),
                    use_rerank=use_rerank, use_multiquery=use_mq,
                )
                reply = engine.answer(query, results, meta,
                                      history=st.session_state.history)
            st.markdown(reply)

            bits = [f"confidence **{meta['confidence']:.2f}**",
                    f"{meta['n_results']} hits",
                    f"{meta['latency_ms']} ms"]
            if meta["typo_fixed"]:
                bits.append(f"corrected to *{meta['corrected']}*")
            st.caption(" · ".join(bits))

            with st.expander(f"Sources ({len(results)})", expanded=True):
                for r in results:
                    st.markdown(
                        f"**{r.title}** — {r.author or 'unknown'}  \n"
                        f"`{r.item_type}` · {r.year or '—'} · score {r.score:.2f} · {r.citation}  \n"
                        f"Subjects: {r.subjects or 'n/a'}  \n"
                        f"📦 {r.copies} copies across {r.locations or 'n/a'}"
                    )
                    st.progress(min(max(r.score, 0.0), 1.0))

        st.session_state.history.append({"role": "assistant", "content": reply})

    if st.session_state.history and st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

with tab_avail:
    st.subheader("Check a specific title's availability")
    title_q = st.text_input("Title (typos are tolerated)")
    if title_q:
        info = engine.lookup_availability(title_q)
        if not info:
            st.warning("No close match found.")
        else:
            st.markdown(f"### {info['title']}")
            st.write(f"by {info['author'] or 'unknown'}  ·  match {info['match_score']:.0%}")
            c1, c2 = st.columns(2)
            c1.metric("Total copies", info["copies"])
            c2.metric("Branches", len(info["locations"]))
            st.write("Locations:", ", ".join(info["locations"]) or "n/a")

with tab_analytics:
    st.subheader("Search history & analytics")
    if not os.path.exists(HISTORY_CSV):
        st.info("No searches logged yet.")
    else:
        h = pd.read_csv(HISTORY_CSV, parse_dates=["ts"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total searches", len(h))
        c2.metric("Avg latency", f"{h['latency_ms'].mean():.0f} ms")
        c3.metric("Avg confidence", f"{h['confidence'].mean():.2f}")
        c4.metric("Typos corrected", f"{(h['typo_fixed'] == True).mean():.0%}")

        st.markdown("**Top queries**")
        st.bar_chart(h["query"].value_counts().head(10))

        st.markdown("**Searches over time**")
        by_day = h.set_index("ts").resample("D").size()
        st.line_chart(by_day)

        st.markdown("**Recent searches**")
        st.dataframe(
            h.sort_values("ts", ascending=False)
             .head(20)[["ts", "query", "corrected", "confidence", "n_results"]],
            use_container_width=True, hide_index=True,
        )
