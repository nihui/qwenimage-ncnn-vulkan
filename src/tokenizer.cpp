// qwen-image implemented with ncnn library

#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <limits>
#include <cstring>

namespace qwenimage {

// ---------------- I/O helpers ----------------
static std::string Trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) --e;
    return s.substr(b, e - b);
}

std::vector<std::string> QwenBpeTokenizer::LoadVocab(const std::string& vocab_path)
{
    std::ifstream ifs(vocab_path, std::ios::binary);
    if (!ifs)
    {
        std::fprintf(stderr, "Failed to open vocab file: %s\n", vocab_path.c_str());
        return {};
    }

    std::vector<std::string> vocab;
    vocab.reserve(151643);
    std::string line;
    while (std::getline(ifs, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        vocab.push_back(line);
    }
    return vocab;
}

#if _WIN32
std::vector<std::string> QwenBpeTokenizer::LoadVocab(const std::wstring& vocab_path) {
    std::ifstream ifs(vocab_path);
    if (!ifs.is_open()) {
        // throw std::runtime_error("Failed to open vocab file: " + vocab_path);
        fwprintf(stderr, L"Failed to open vocab file: %ls\n", vocab_path.c_str());
    }
    std::vector<std::string> vocab;
    vocab.reserve(151643);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        vocab.push_back(line);
    }
    return vocab;
}
#endif

std::unordered_map<std::string, int> QwenBpeTokenizer::BuildTokenToId(const std::vector<std::string>& id_to_token) {
    std::unordered_map<std::string, int> m;
    m.reserve(id_to_token.size() * 2);
    for (size_t i = 0; i < id_to_token.size(); ++i) {
        m.emplace(id_to_token[i], static_cast<int>(i));
    }
    return m;
}

std::unordered_map<std::string, int> QwenBpeTokenizer::LoadMergesRank(const std::string& merges_path) {
    std::ifstream ifs(merges_path);
    if (!ifs.is_open()) {
        // throw std::runtime_error("Failed to open merges file: " + merges_path);
        fprintf(stderr, "Failed to open merges file: %s\n", merges_path.c_str());
    }
    std::unordered_map<std::string, int> merges;
    merges.reserve(50000);
    std::string line;
    int rank = 0;
    while (std::getline(ifs, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string a, b;
        if (!(iss >> a >> b)) continue;
        merges.emplace(PairKey(a, b), rank++);
    }
    return merges;
}

#if _WIN32
std::unordered_map<std::string, int> QwenBpeTokenizer::LoadMergesRank(const std::wstring& merges_path) {
    std::ifstream ifs(merges_path);
    if (!ifs.is_open()) {
        // throw std::runtime_error("Failed to open merges file: " + merges_path);
        fwprintf(stderr, L"Failed to open merges file: %ls\n", merges_path.c_str());
    }
    std::unordered_map<std::string, int> merges;
    merges.reserve(50000);
    std::string line;
    int rank = 0;
    while (std::getline(ifs, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string a, b;
        if (!(iss >> a >> b)) continue;
        merges.emplace(PairKey(a, b), rank++);
    }
    return merges;
}
#endif

// ---------------- UTF-8 utilities ----------------

bool QwenBpeTokenizer::IsAsciiSpace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool QwenBpeTokenizer::IsUnicodeSpace(uint32_t cp) {
    if (cp <= 0x7F) return IsAsciiSpace(static_cast<unsigned char>(cp));
    switch (cp) {
        case 0x00A0: case 0x1680:
        case 0x2000: case 0x2001: case 0x2002: case 0x2003:
        case 0x2004: case 0x2005: case 0x2006: case 0x2007:
        case 0x2008: case 0x2009: case 0x200A:
        case 0x2028: case 0x2029:
        case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return false;
    }
}

bool QwenBpeTokenizer::NextUtf8(const std::string& s, size_t& i, uint32_t& cp, size_t& cp_len) {
    if (i >= s.size()) return false;
    unsigned char c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80) {
        cp = c0; cp_len = 1; ++i; return true;
    } else if ((c0 >> 5) == 0x6) {
        if (i + 1 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i+1]);
        if ((c1 & 0xC0) != 0x80) return false;
        cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        cp_len = 2; i += 2; return true;
    } else if ((c0 >> 4) == 0xE) {
        if (i + 2 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i+1]);
        unsigned char c2 = static_cast<unsigned char>(s[i+2]);
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80)) return false;
        cp = ((c0 & 0x0F) << 12) |
             ((c1 & 0x3F) << 6) |
             (c2 & 0x3F);
        cp_len = 3; i += 3; return true;
    } else if ((c0 >> 3) == 0x1E) {
        if (i + 3 >= s.size()) return false;
        unsigned char c1 = static_cast<unsigned char>(s[i+1]);
        unsigned char c2 = static_cast<unsigned char>(s[i+2]);
        unsigned char c3 = static_cast<unsigned char>(s[i+3]);
        if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80) || ((c3 & 0xC0) != 0x80)) return false;
        cp = ((c0 & 0x07) << 18) |
             ((c1 & 0x3F) << 12) |
             ((c2 & 0x3F) << 6) |
             (c3 & 0x3F);
        cp_len = 4; i += 4; return true;
    }
    return false;
}

std::vector<std::string> QwenBpeTokenizer::Utf8Chars(const std::string& s) {
    std::vector<std::string> chars;
    chars.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        size_t start = i;
        uint32_t cp; size_t cp_len;
        if (!NextUtf8(s, i, cp, cp_len)) break;
        chars.emplace_back(s.substr(start, i - start));
    }
    return chars;
}

std::vector<std::string> QwenBpeTokenizer::PretokenizeSentencePiece(const std::string& text)
{
    // Qwen2 uses the byte-level path.  Keep the fallback path link-complete
    // for the shared tokenizer implementation.
    if (text.empty())
        return {};
    return {text};
}

// ---------------- Byte Encoder/Decoder (New Logic) ----------------

void QwenBpeTokenizer::InitByteMaps() {
    byte_encoder_.resize(256, 0);
    byte_decoder_.clear();

    // Logic based on GPT-2 byte encoder (map printable chars to themselves, others to 256+)
    auto is_printable = [](int b) {
        return (b >= '!' && b <= '~')     // '!' to '~'
                || (b >= 161 && b <= 172)  // '¡' to '¬'
                || (b >= 174 && b <= 255); // '®' to 'ÿ'
    };

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (is_printable(b)) {
            byte_encoder_[b] = static_cast<uint32_t>(b);
        } else {
            byte_encoder_[b] = static_cast<uint32_t>(256 + n);
            n++;
        }
        // Build decoder map
        byte_decoder_[byte_encoder_[b]] = static_cast<uint8_t>(b);
    }
}

std::string QwenBpeTokenizer::ByteEncode(const std::string& text) const {
    std::string out;
    out.reserve(text.size() * 2);

    for (unsigned char b : text) {
        uint32_t cp = byte_encoder_[b];
        // Append cp as UTF-8
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>((cp >> 6) | 0xC0);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        } else if (cp < 0x10000) {
            out += static_cast<char>((cp >> 12) | 0xE0);
            out += static_cast<char>(((cp >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        } else {
            out += static_cast<char>((cp >> 18) | 0xF0);
            out += static_cast<char>(((cp >> 12) & 0x3F) | 0x80);
            out += static_cast<char>(((cp >> 6) & 0x3F) | 0x80);
            out += static_cast<char>((cp & 0x3F) | 0x80);
        }
    }
    return out;
}

std::string QwenBpeTokenizer::ByteDecode(const std::string& text) const {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    uint32_t cp; size_t len;
    while (i < text.size()) {
        if (!NextUtf8(text, i, cp, len)) break;

        auto it = byte_decoder_.find(cp);
        if (it != byte_decoder_.end()) {
            out += static_cast<char>(it->second);
        } else {
            // If codepoint is not in byte map, usually we ignore it or keep it
            // if it's part of a special token that wasn't filtered.
            // Here we ignore unmapped chars to ensure pure byte restoration.
        }
    }
    return out;
}

// ---------------- Pretokenizer ----------------
std::vector<std::string> QwenBpeTokenizer::PretokenizeByteLevel(const std::string& text)
{
    auto is_letter = [](uint32_t cp) {
        return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')
            || (cp >= 0x00c0 && cp <= 0x02ff)
            || (cp >= 0x0370 && cp <= 0x052f)
            || (cp >= 0x1e00 && cp <= 0x1eff)
            || (cp >= 0x3040 && cp <= 0x30ff)
            || (cp >= 0x3400 && cp <= 0x4dbf)
            || (cp >= 0x4e00 && cp <= 0x9fff)
            || (cp >= 0xa000 && cp <= 0xa4cf)
            || (cp >= 0xac00 && cp <= 0xd7af)
            || (cp >= 0xf900 && cp <= 0xfaff)
            || (cp >= 0x10000 && cp <= 0x1efff);
    };
    auto is_number = [](uint32_t cp) {
        return (cp >= '0' && cp <= '9')
            || (cp >= 0x0660 && cp <= 0x0669)
            || (cp >= 0x06f0 && cp <= 0x06f9)
            || (cp >= 0x0966 && cp <= 0x096f)
            || (cp >= 0xff10 && cp <= 0xff19);
    };
    auto is_newline = [](uint32_t cp) {
        return cp == '\r' || cp == '\n';
    };
    auto lower_ascii = [](char c) {
        return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c;
    };
    auto next_cp = [&](size_t p, uint32_t& cp, size_t& end) {
        size_t q = p;
        if (!NextUtf8(text, q, cp, end))
            return false;
        end = q;
        return true;
    };
    auto append_range = [&](std::vector<std::string>& out, size_t a, size_t b) {
        if (b > a)
            out.emplace_back(text.substr(a, b - a));
    };

    std::vector<std::string> out;
    size_t pos = 0;
    while (pos < text.size())
    {
        // GPT-2's contraction alternative has priority over the letter rule.
        bool contraction = false;
        if (text[pos] == '\'')
        {
            static const char* suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            for (const char* suffix : suffixes)
            {
                const size_t n = std::strlen(suffix);
                if (pos + n > text.size())
                    continue;
                bool match = true;
                for (size_t i = 0; i < n; i++)
                    if (lower_ascii(text[pos + i]) != lower_ascii(suffix[i]))
                        match = false;
                if (match)
                {
                    append_range(out, pos, pos + n);
                    pos += n;
                    contraction = true;
                    break;
                }
            }
        }
        if (contraction)
            continue;

        uint32_t cp = 0;
        size_t next = pos;
        if (!next_cp(pos, cp, next))
            break;

        // [^... ]?\p{L}+
        size_t p = pos;
        uint32_t qcp = cp;
        size_t qnext = next;
        if (!is_newline(cp) && !is_letter(cp) && !is_number(cp))
        {
            size_t after = qnext;
            uint32_t next_letter = 0;
            if (next_cp(qnext, next_letter, after) && is_letter(next_letter))
            {
                p = qnext;
                qcp = next_letter;
                qnext = after;
            }
        }
        if (is_letter(qcp))
        {
            size_t end = qnext;
            while (end < text.size())
            {
                uint32_t c = 0;
                size_t e = end;
                if (!next_cp(end, c, e) || !is_letter(c))
                    break;
                end = e;
            }
            append_range(out, pos, end);
            pos = end;
            continue;
        }

        // \p{N}
        if (is_number(cp))
        {
            append_range(out, pos, next);
            pos = next;
            continue;
        }

        // ?[^\s\p{L}\p{N}]+[\r\n]*
        size_t punct_begin = pos;
        size_t punct = pos;
        if (cp == ' ')
        {
            uint32_t after_space = 0;
            size_t after = next;
            if (next_cp(next, after_space, after)
                && !IsUnicodeSpace(after_space)
                && !is_letter(after_space) && !is_number(after_space))
                punct = next;
        }
        uint32_t pc = 0;
        size_t pe = punct;
        if (next_cp(punct, pc, pe)
            && !IsUnicodeSpace(pc) && !is_letter(pc) && !is_number(pc)
            && !is_newline(pc))
        {
            size_t end = pe;
            while (end < text.size())
            {
                uint32_t c = 0;
                size_t e = end;
                if (!next_cp(end, c, e)
                    || IsUnicodeSpace(c) || is_letter(c) || is_number(c)
                    || is_newline(c))
                    break;
                end = e;
            }
            while (end < text.size())
            {
                uint32_t c = 0;
                size_t e = end;
                if (!next_cp(end, c, e) || !is_newline(c))
                    break;
                end = e;
            }
            append_range(out, punct_begin, end);
            pos = end;
            continue;
        }

        // \s*[\r\n]+, or a whitespace run.
        if (IsUnicodeSpace(cp))
        {
            size_t end = pos;
            bool newline = false;
            while (end < text.size())
            {
                uint32_t c = 0;
                size_t e = end;
                if (!next_cp(end, c, e) || !IsUnicodeSpace(c))
                    break;
                newline = newline || is_newline(c);
                end = e;
            }
            append_range(out, pos, end);
            pos = end;
            continue;
        }

        append_range(out, pos, next);
        pos = next;
    }
    return out;
}

// ---------------- BPE core ----------------

std::string QwenBpeTokenizer::PairKey(const std::string& a, const std::string& b) {
    std::string key;
    key.reserve(a.size() + 1 + b.size());
    key.append(a);
    key.push_back('\t');
    key.append(b);
    return key;
}

const std::vector<std::string>& QwenBpeTokenizer::BpeForPieceCached(const std::string& piece) const {
    {
        std::lock_guard<std::mutex> g(cache_mu_);
        auto it = bpe_cache_.find(piece);
        if (it != bpe_cache_.end()) return it->second;
    }
    auto tokens = BpeForPiece(piece);
    {
        std::lock_guard<std::mutex> g(cache_mu_);
        auto result = bpe_cache_.emplace(piece, std::move(tokens));
        return result.first->second;
    }
}

std::vector<std::string> QwenBpeTokenizer::BpeForPiece(const std::string& piece) const {
    std::vector<std::string> symbols = Utf8Chars(piece);
    if (symbols.size() <= 1) return symbols;

    while (symbols.size() >= 2) {
        int best_rank = std::numeric_limits<int>::max();
        int best_i = -1;

        for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
            std::string key = PairKey(symbols[i], symbols[i + 1]);
            auto it = merges_rank_.find(key);
            if (it != merges_rank_.end()) {
                int r = it->second;
                if (r < best_rank) {
                    best_rank = r;
                    best_i = i;
                }
            }
        }
        if (best_i < 0) break;

        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }
    return symbols;
}

void QwenBpeTokenizer::TokensToIds(const std::vector<std::string>& tokens, std::vector<int>& out) const {
    for (const auto& t : tokens) {
        auto it = token_to_id_.find(t);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else if (fallback_to_chars_) {
            auto chars = Utf8Chars(t);
            for (const auto& ch : chars) {
                auto it2 = token_to_id_.find(ch);
                if (it2 != token_to_id_.end()) {
                    out.push_back(it2->second);
                } else if (special_ids_.unk_id >= 0) {
                    out.push_back(special_ids_.unk_id);
                }
            }
        } else if (special_ids_.unk_id >= 0) {
            out.push_back(special_ids_.unk_id);
        }
    }
}

// ---------------- Move semantics ----------------
QwenBpeTokenizer::QwenBpeTokenizer(QwenBpeTokenizer&& other) noexcept
    : id_to_token_(std::move(other.id_to_token_)),
      token_to_id_(std::move(other.token_to_id_)),
      merges_rank_(std::move(other.merges_rank_)),
      special_ids_(other.special_ids_),
      fallback_to_chars_(other.fallback_to_chars_),
      additional_special_tokens_(std::move(other.additional_special_tokens_)),
      additional_special_token_ids_(std::move(other.additional_special_token_ids_)),
      additional_special_token_to_id_(std::move(other.additional_special_token_to_id_)),
      additional_special_id_set_(std::move(other.additional_special_id_set_)),
      use_byte_encoder_(other.use_byte_encoder_),
      byte_encoder_(std::move(other.byte_encoder_)),
      byte_decoder_(std::move(other.byte_decoder_)) {
    std::lock_guard<std::mutex> lk(other.cache_mu_);
    bpe_cache_ = std::move(other.bpe_cache_);
}

QwenBpeTokenizer& QwenBpeTokenizer::operator=(QwenBpeTokenizer&& other) noexcept {
    if (this != &other) {
        std::lock(cache_mu_, other.cache_mu_);
        std::lock_guard<std::mutex> lock1(cache_mu_, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.cache_mu_, std::adopt_lock);

        id_to_token_ = std::move(other.id_to_token_);
        token_to_id_ = std::move(other.token_to_id_);
        merges_rank_ = std::move(other.merges_rank_);
        special_ids_ = other.special_ids_;
        fallback_to_chars_ = other.fallback_to_chars_;
        additional_special_tokens_ = std::move(other.additional_special_tokens_);
        additional_special_token_ids_ = std::move(other.additional_special_token_ids_);
        additional_special_token_to_id_ = std::move(other.additional_special_token_to_id_);
        additional_special_id_set_ = std::move(other.additional_special_id_set_);
        use_byte_encoder_ = other.use_byte_encoder_;
        byte_encoder_ = std::move(other.byte_encoder_);
        byte_decoder_ = std::move(other.byte_decoder_);
        bpe_cache_ = std::move(other.bpe_cache_);
    }
    return *this;
}

// ---------------- Public APIs ----------------
QwenBpeTokenizer QwenBpeTokenizer::LoadFromFiles(const std::string& vocab_path,
                                         const std::string& merges_path,
                                         const SpecialTokensConfig& spec,
                                         bool add_special_if_missing,
                                         bool fallback_to_chars,
                                         bool use_byte_encoder) {
    QwenBpeTokenizer tok;
    tok.id_to_token_ = LoadVocab(vocab_path);
    if (tok.id_to_token_.empty()) {
        // throw std::runtime_error("Vocab is empty: " + vocab_path);
        fprintf(stderr, "Vocab is empty: %s\n", vocab_path.c_str());
    }
    tok.token_to_id_ = BuildTokenToId(tok.id_to_token_);
    tok.merges_rank_ = LoadMergesRank(merges_path);
    tok.fallback_to_chars_ = fallback_to_chars;
    tok.use_byte_encoder_ = use_byte_encoder;

    if (tok.use_byte_encoder_) {
        tok.InitByteMaps();
    }

    tok.EnsureSpecialTokens(spec, add_special_if_missing);
    return std::move(tok);
}

#if _WIN32
QwenBpeTokenizer QwenBpeTokenizer::LoadFromFiles(const std::wstring& vocab_path,
                                         const std::wstring& merges_path,
                                         const SpecialTokensConfig& spec,
                                         bool add_special_if_missing,
                                         bool fallback_to_chars,
                                         bool use_byte_encoder) {
    QwenBpeTokenizer tok;
    tok.id_to_token_ = LoadVocab(vocab_path);
    if (tok.id_to_token_.empty()) {
        // throw std::runtime_error("Vocab is empty: " + vocab_path);
        fwprintf(stderr, L"Vocab is empty: %ls\n", vocab_path.c_str());
    }
    tok.token_to_id_ = BuildTokenToId(tok.id_to_token_);
    tok.merges_rank_ = LoadMergesRank(merges_path);
    tok.fallback_to_chars_ = fallback_to_chars;
    tok.use_byte_encoder_ = use_byte_encoder;

    if (tok.use_byte_encoder_) {
        tok.InitByteMaps();
    }

    tok.EnsureSpecialTokens(spec, add_special_if_missing);
    return std::move(tok);
}
#endif

void QwenBpeTokenizer::EnsureSpecialTokens(const SpecialTokensConfig& spec, bool add_if_missing) {
    auto ensure = [&](const std::string& name, int& id_slot) {
        if (name.empty()) { id_slot = -1; return; }
        auto it = token_to_id_.find(name);
        if (it != token_to_id_.end()) {
            id_slot = it->second;
            return;
        }
        if (!add_if_missing) {
            id_slot = -1;
            return;
        }
        id_slot = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(name);
        token_to_id_.emplace(name, id_slot);
    };

    ensure(spec.bos_token, special_ids_.bos_id);
    ensure(spec.eos_token, special_ids_.eos_id);
    ensure(spec.unk_token, special_ids_.unk_id);
    ensure(spec.sep_token, special_ids_.sep_id);
    ensure(spec.pad_token, special_ids_.pad_id);
    ensure(spec.cls_token, special_ids_.cls_id);
    ensure(spec.mask_token, special_ids_.mask_id);
}

void QwenBpeTokenizer::AddAdditionalSpecialToken(const std::string& token,
                                             bool add_if_missing) {
    if (token.empty()) return;

    auto it_exist = additional_special_token_to_id_.find(token);
    if (it_exist != additional_special_token_to_id_.end()) {
        return;
    }

    int id = -1;
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        id = it->second;
    } else if (add_if_missing) {
        id = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(token);
        token_to_id_.emplace(token, id);
    } else {
        return;
    }

    additional_special_tokens_.push_back(token);
    additional_special_token_ids_.push_back(id);
    additional_special_token_to_id_.emplace(token, id);
    additional_special_id_set_.insert(id);
}

void QwenBpeTokenizer::SetAdditionalSpecialTokens(const std::vector<std::string>& tokens,
                                              bool add_if_missing) {
    additional_special_tokens_.clear();
    additional_special_token_ids_.clear();
    additional_special_token_to_id_.clear();
    additional_special_id_set_.clear();

    for (const auto& t : tokens) {
        AddAdditionalSpecialToken(t, add_if_missing);
    }
}

std::vector<int> QwenBpeTokenizer::encode(const std::string& text,
                                      bool add_bos,
                                      bool add_eos,
                                      bool add_cls,
                                      bool add_sep) const {
    std::vector<int> ids;
    ids.reserve(text.size() / 2 + 8);

    if (add_cls && special_ids_.cls_id >= 0) ids.push_back(special_ids_.cls_id);
    if (add_bos && special_ids_.bos_id >= 0) ids.push_back(special_ids_.bos_id);

    std::string buffer;
    buffer.reserve(text.size());

    auto flush_buffer = [&]() {
        if (buffer.empty()) return;

        if (use_byte_encoder_) {
            // Qwen2 uses the GPT-2 regex pre-tokenizer followed by
            // byte-level encoding and BPE.
            for (const std::string& piece : PretokenizeByteLevel(buffer))
            {
                const std::string encoded_s = ByteEncode(piece);
                const auto& toks = BpeForPieceCached(encoded_s);
                TokensToIds(toks, ids);
            }
        } else {
            auto pieces = PretokenizeSentencePiece(buffer);
            for (const auto& p : pieces) {
                const auto& toks = BpeForPieceCached(p);
                TokensToIds(toks, ids);
            }
        }
        buffer.clear();
    };

    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        int matched_index = -1;
        size_t matched_len = 0;

        if (!additional_special_tokens_.empty()) {
            for (size_t k = 0; k < additional_special_tokens_.size(); ++k) {
                const std::string& sp = additional_special_tokens_[k];
                if (sp.empty()) continue;
                size_t len = sp.size();
                if (len <= matched_len) continue;
                if (i + len <= n && text.compare(i, len, sp) == 0) {
                    matched_index = static_cast<int>(k);
                    matched_len = len;
                }
            }
        }

        if (matched_index >= 0) {
            flush_buffer();
            ids.push_back(additional_special_token_ids_[matched_index]);
            i += matched_len;
            continue;
        }

        buffer.push_back(text[i]);
        ++i;
    }

    flush_buffer();

    if (add_sep && special_ids_.sep_id >= 0) ids.push_back(special_ids_.sep_id);
    if (add_eos && special_ids_.eos_id >= 0) ids.push_back(special_ids_.eos_id);
    return ids;
}

std::string QwenBpeTokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    std::string s;
    s.reserve(ids.size() * 3);
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
        const std::string& tok = id_to_token_[id];

        bool is_special =
            id == special_ids_.bos_id || id == special_ids_.eos_id ||
            id == special_ids_.unk_id || id == special_ids_.sep_id ||
            id == special_ids_.pad_id || id == special_ids_.cls_id ||
            id == special_ids_.mask_id ||
            (additional_special_id_set_.find(id) != additional_special_id_set_.end());

        if (skip_special_tokens && is_special) continue;

        if (tok == "\\t")
            s += '\t';
        else if (tok == "\n")
            s += '\n';
        else if (tok == "\r")
            s += '\r';
        else
            s += tok;
    }

    if (!s.empty()) {
        // Branch based on encoding mode
        if (use_byte_encoder_) {
            return ByteDecode(s);
        } else {
            std::string out;
            out.reserve(s.size());
            size_t i = 0;
            while (i < s.size()) {
                uint32_t cp; size_t len;
                if (!NextUtf8(s, i, cp, len)) break;
                if (cp == 0x2581) {
                    out.push_back(' ');
                } else {
                    out.append(s, i - len, len);
                }
            }
            size_t b = 0;
            while (b < out.size() && out[b] == ' ') ++b;
            if (b > 0) out.erase(0, b);
            return out;
        }
    }
    return s;
}

} // namespace qwenimage
