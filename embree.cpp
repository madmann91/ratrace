// avxb.h
namespace embree
{
  /*! 8-wide AVX bool type. */
  struct avxb
  {
    typedef avxb Mask;         // mask type for us
    enum   { size = 8 };       // number of SIMD elements
    union  {                   // data
      __m256 m256; 
      struct { __m128 l,h; }; 
      int32 v[8]; 
    };  

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline avxb           () {}
    __forceinline avxb           ( const avxb& a ) { m256 = a.m256; }
    __forceinline avxb& operator=( const avxb& a ) { m256 = a.m256; return *this; }

    __forceinline avxb( const __m256 a ) : m256(a) {}
    __forceinline operator const __m256&( void ) const { return m256; }
    __forceinline operator const __m256i( void ) const { return _mm256_castps_si256(m256); }
    __forceinline operator const __m256d( void ) const { return _mm256_castps_pd(m256); }

    __forceinline avxb ( const int a ) 
    {
      assert(a >= 0 && a <= 255);
#if defined (__AVX2__)
      const __m256i mask = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1);
      const __m256i b = _mm256_set1_epi32(a);
      const __m256i c = _mm256_and_si256(b,mask);
      m256 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(c,mask));
#else
      l = _mm_lookupmask_ps[a & 0xF];
      h = _mm_lookupmask_ps[a >> 4];
#endif
    }
    
    __forceinline avxb ( const  sseb& a                ) : m256(_mm256_insertf128_ps(_mm256_castps128_ps256(a),a,1)) {}
    __forceinline avxb ( const  sseb& a, const  sseb& b) : m256(_mm256_insertf128_ps(_mm256_castps128_ps256(a),b,1)) {}
    __forceinline avxb ( const __m128 a, const __m128 b) : l(a), h(b) {}

    __forceinline avxb ( bool a ) : m256(avxb(sseb(a), sseb(a))) {}
    __forceinline avxb ( bool a, bool b) : m256(avxb(sseb(a), sseb(b))) {}
    __forceinline avxb ( bool a, bool b, bool c, bool d) : m256(avxb(sseb(a,b), sseb(c,d))) {}
    __forceinline avxb ( bool a, bool b, bool c, bool d, bool e, bool f, bool g, bool h ) : m256(avxb(sseb(a,b,c,d), sseb(e,f,g,h))) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline avxb( FalseTy ) : m256(_mm256_setzero_ps()) {}
    __forceinline avxb( TrueTy  ) : m256(_mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline bool   operator []( const size_t i ) const { assert(i < 8); return (_mm256_movemask_ps(m256) >> i) & 1; }
    __forceinline int32& operator []( const size_t i )       { assert(i < 8); return v[i]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const avxb operator !( const avxb& a ) { return _mm256_xor_ps(a, avxb(True)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const avxb operator &( const avxb& a, const avxb& b ) { return _mm256_and_ps(a, b); }
  __forceinline const avxb operator |( const avxb& a, const avxb& b ) { return _mm256_or_ps (a, b); }
  __forceinline const avxb operator ^( const avxb& a, const avxb& b ) { return _mm256_xor_ps(a, b); }

  __forceinline avxb operator &=( avxb& a, const avxb& b ) { return a = a & b; }
  __forceinline avxb operator |=( avxb& a, const avxb& b ) { return a = a | b; }
  __forceinline avxb operator ^=( avxb& a, const avxb& b ) { return a = a ^ b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const avxb operator !=( const avxb& a, const avxb& b ) { return _mm256_xor_ps(a, b); }
  __forceinline const avxb operator ==( const avxb& a, const avxb& b ) { return _mm256_xor_ps(_mm256_xor_ps(a,b),avxb(True)); }

  __forceinline const avxb select( const avxb& mask, const avxb& t, const avxb& f ) { 
    return _mm256_blendv_ps(f, t, mask); 
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline avxb unpacklo( const avxb& a, const avxb& b ) { return _mm256_unpacklo_ps(a.m256, b.m256); }
  __forceinline avxb unpackhi( const avxb& a, const avxb& b ) { return _mm256_unpackhi_ps(a.m256, b.m256); }

  template<size_t i> __forceinline const avxb shuffle( const avxb& a ) {
    return _mm256_permute_ps(a, _MM_SHUFFLE(i, i, i, i));
  }

  template<size_t i0, size_t i1> __forceinline const avxb shuffle( const avxb& a ) {
    return _mm256_permute2f128_ps(a, a, (i1 << 4) | (i0 << 0));
  }

  template<size_t i0, size_t i1> __forceinline const avxb shuffle( const avxb& a,  const avxb& b) {
    return _mm256_permute2f128_ps(a, b, (i1 << 4) | (i0 << 0));
  }

  template<size_t i0, size_t i1, size_t i2, size_t i3> __forceinline const avxb shuffle( const avxb& a ) {
    return _mm256_permute_ps(a, _MM_SHUFFLE(i3, i2, i1, i0));
  }

  template<size_t i0, size_t i1, size_t i2, size_t i3> __forceinline const avxb shuffle( const avxb& a, const avxb& b ) {
    return _mm256_shuffle_ps(a, b, _MM_SHUFFLE(i3, i2, i1, i0));
  }

  template<> __forceinline const avxb shuffle<0, 0, 2, 2>( const avxb& b ) { return _mm256_moveldup_ps(b); }
  template<> __forceinline const avxb shuffle<1, 1, 3, 3>( const avxb& b ) { return _mm256_movehdup_ps(b); }
  template<> __forceinline const avxb shuffle<0, 1, 0, 1>( const avxb& b ) { return _mm256_castpd_ps(_mm256_movedup_pd(_mm256_castps_pd(b))); }

  template<size_t i> __forceinline const avxb insert (const avxb& a, const sseb& b) { return _mm256_insertf128_ps (a,b,i); }
  template<size_t i> __forceinline const sseb extract(const avxb& a               ) { return _mm256_extractf128_ps(a  ,i); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reduction Operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool reduce_and( const avxb& a ) { return _mm256_movemask_ps(a) == (unsigned int)0xff; }
  __forceinline bool reduce_or ( const avxb& a ) { return !_mm256_testz_ps(a,a); }

  __forceinline bool all       ( const avxb& a ) { return _mm256_movemask_ps(a) == (unsigned int)0xff; }
  __forceinline bool any       ( const avxb& a ) { return !_mm256_testz_ps(a,a); }
  __forceinline bool none      ( const avxb& a ) { return _mm256_testz_ps(a,a) != 0; }

  __forceinline bool all       ( const avxb& valid, const avxb& b ) { return all(!valid | b); }
  __forceinline bool any       ( const avxb& valid, const avxb& b ) { return any( valid & b); }
  __forceinline bool none      ( const avxb& valid, const avxb& b ) { return none(valid & b); }

  __forceinline unsigned int movemask( const avxb& a ) { return _mm256_movemask_ps(a); }
  __forceinline size_t       popcnt  ( const avxb& a ) { return __popcnt(_mm256_movemask_ps(a)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  inline std::ostream& operator<<(std::ostream& cout, const avxb& a) {
    return cout << "<" << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << ", "
                       << a[4] << ", " << a[5] << ", " << a[6] << ", " << a[7] << ">";
  }
}

// avxf.h
namespace embree
{
  /*! 8-wide AVX float type. */
  struct avxf
  {
    typedef avxb Mask;    // mask type for us
    typedef avxi Int ;    // int type for us
    enum   { size = 8 };  // number of SIMD elements
    union { __m256 m256; float v[8]; }; // data

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline avxf           ( ) {}
    __forceinline avxf           ( const avxf& other ) { m256 = other.m256; }
    __forceinline avxf& operator=( const avxf& other ) { m256 = other.m256; return *this; }

    __forceinline avxf( const __m256  a ) : m256(a) {}
    __forceinline operator const __m256&( void ) const { return m256; }
    __forceinline operator       __m256&( void )       { return m256; }

    __forceinline explicit avxf( const ssef& a                ) : m256(_mm256_insertf128_ps(_mm256_castps128_ps256(a),a,1)) {}
    __forceinline          avxf( const ssef& a, const ssef& b ) : m256(_mm256_insertf128_ps(_mm256_castps128_ps256(a),b,1)) {}

    static __forceinline avxf load( const void* const ptr ) { return *(__m256*)ptr; }

    __forceinline explicit avxf( const char* const a ) : m256(_mm256_loadu_ps((const float*)a)) {}
    __forceinline          avxf( const float&       a ) : m256(_mm256_broadcast_ss(&a)) {}
    __forceinline          avxf( float a, float b) : m256(_mm256_set_ps(b, a, b, a, b, a, b, a)) {}
    __forceinline          avxf( float a, float b, float c, float d ) : m256(_mm256_set_ps(d, c, b, a, d, c, b, a)) {}
    __forceinline          avxf( float a, float b, float c, float d, float e, float f, float g, float h ) : m256(_mm256_set_ps(h, g, f, e, d, c, b, a)) {}

    __forceinline explicit avxf( const __m256i a ) : m256(_mm256_cvtepi32_ps(a)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline avxf( ZeroTy   ) : m256(_mm256_setzero_ps()) {}
    __forceinline avxf( OneTy    ) : m256(_mm256_set1_ps(1.0f)) {}
    __forceinline avxf( PosInfTy ) : m256(_mm256_set1_ps(pos_inf)) {}
    __forceinline avxf( NegInfTy ) : m256(_mm256_set1_ps(neg_inf)) {}
    __forceinline avxf( StepTy   ) : m256(_mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f)) {}
    __forceinline avxf( NaNTy    ) : m256(_mm256_set1_ps(nan)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline avxf broadcast( const void* const a ) { 
      return _mm256_broadcast_ss((float*)a); 
    }

    static __forceinline avxf load( const unsigned char* const ptr ) { 
#if defined(__AVX2__)
      return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)ptr)));
#else
      return avxf(ssef::load(ptr),ssef::load(ptr+4));
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline const float& operator []( const size_t i ) const { assert(i < 8); return v[i]; }
    __forceinline       float& operator []( const size_t i )       { assert(i < 8); return v[i]; }
  };


  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const avxf cast      (const avxi& a   ) { return _mm256_castsi256_ps(a); }
  __forceinline const avxi cast      (const avxf& a   ) { return _mm256_castps_si256(a); }
  __forceinline const avxf operator +( const avxf& a ) { return a; }
  __forceinline const avxf operator -( const avxf& a ) { 
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)); 
    return _mm256_xor_ps(a.m256, mask); 
  }
  __forceinline const avxf abs  ( const avxf& a ) { 
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    return _mm256_and_ps(a.m256, mask); 
  }
  __forceinline const avxf sign    ( const avxf& a ) { return _mm256_blendv_ps(avxf(one), -avxf(one), _mm256_cmp_ps(a, avxf(zero), _CMP_NGE_UQ )); }
  __forceinline const avxf signmsk ( const avxf& a ) { return _mm256_and_ps(a.m256,_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000))); }

  __forceinline const avxf rcp  ( const avxf& a ) { 
    const avxf r = _mm256_rcp_ps(a.m256); 
    return _mm256_sub_ps(_mm256_add_ps(r, r), _mm256_mul_ps(_mm256_mul_ps(r, r), a)); 
  }
  __forceinline const avxf sqr  ( const avxf& a ) { return _mm256_mul_ps(a,a); }
  __forceinline const avxf sqrt ( const avxf& a ) { return _mm256_sqrt_ps(a.m256); }
  __forceinline const avxf rsqrt( const avxf& a ) { 
    const avxf r = _mm256_rsqrt_ps(a.m256);
    return _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(1.5f), r), _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(a, _mm256_set1_ps(-0.5f)), r), _mm256_mul_ps(r, r))); 
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const avxf operator +( const avxf& a, const avxf& b ) { return _mm256_add_ps(a.m256, b.m256); }
  __forceinline const avxf operator +( const avxf& a, const float b ) { return a + avxf(b); }
  __forceinline const avxf operator +( const float a, const avxf& b ) { return avxf(a) + b; }

  __forceinline const avxf operator -( const avxf& a, const avxf& b ) { return _mm256_sub_ps(a.m256, b.m256); }
  __forceinline const avxf operator -( const avxf& a, const float b ) { return a - avxf(b); }
  __forceinline const avxf operator -( const float a, const avxf& b ) { return avxf(a) - b; }

  __forceinline const avxf operator *( const avxf& a, const avxf& b ) { return _mm256_mul_ps(a.m256, b.m256); }
  __forceinline const avxf operator *( const avxf& a, const float b ) { return a * avxf(b); }
  __forceinline const avxf operator *( const float a, const avxf& b ) { return avxf(a) * b; }

  __forceinline const avxf operator /( const avxf& a, const avxf& b ) { return _mm256_div_ps(a.m256, b.m256); }
  __forceinline const avxf operator /( const avxf& a, const float b ) { return a / avxf(b); }
  __forceinline const avxf operator /( const float a, const avxf& b ) { return avxf(a) / b; }

  __forceinline const avxf operator^( const avxf& a, const avxf& b ) { return _mm256_xor_ps(a.m256,b.m256); }
  __forceinline const avxf operator^( const avxf& a, const avxi& b ) { return _mm256_xor_ps(a.m256,_mm256_castsi256_ps(b.m256)); }

  __forceinline const avxf operator&( const avxf& a, const avxf& b ) { return _mm256_and_ps(a.m256,b.m256); }

  __forceinline const avxf min( const avxf& a, const avxf& b ) { return _mm256_min_ps(a.m256, b.m256); }
  __forceinline const avxf min( const avxf& a, const float b ) { return _mm256_min_ps(a.m256, avxf(b)); }
  __forceinline const avxf min( const float a, const avxf& b ) { return _mm256_min_ps(avxf(a), b.m256); }

  __forceinline const avxf max( const avxf& a, const avxf& b ) { return _mm256_max_ps(a.m256, b.m256); }
  __forceinline const avxf max( const avxf& a, const float b ) { return _mm256_max_ps(a.m256, avxf(b)); }
  __forceinline const avxf max( const float a, const avxf& b ) { return _mm256_max_ps(avxf(a), b.m256); }

#if defined (__AVX2__)
    __forceinline avxf mini(const avxf& a, const avxf& b) {
      const avxi ai = _mm256_castps_si256(a);
      const avxi bi = _mm256_castps_si256(b);
      const avxi ci = _mm256_min_epi32(ai,bi);
      return _mm256_castsi256_ps(ci);
    }
    __forceinline avxf maxi(const avxf& a, const avxf& b) {
      const avxi ai = _mm256_castps_si256(a);
      const avxi bi = _mm256_castps_si256(b);
      const avxi ci = _mm256_max_epi32(ai,bi);
      return _mm256_castsi256_ps(ci);
    }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX2__)
  __forceinline const avxf madd  ( const avxf& a, const avxf& b, const avxf& c) { return _mm256_fmadd_ps(a,b,c); }
  __forceinline const avxf msub  ( const avxf& a, const avxf& b, const avxf& c) { return _mm256_fmsub_ps(a,b,c); }
  __forceinline const avxf nmadd ( const avxf& a, const avxf& b, const avxf& c) { return _mm256_fnmadd_ps(a,b,c); }
  __forceinline const avxf nmsub ( const avxf& a, const avxf& b, const avxf& c) { return _mm256_fnmsub_ps(a,b,c); }
#else
  __forceinline const avxf madd  ( const avxf& a, const avxf& b, const avxf& c) { return a*b+c; }
  __forceinline const avxf msub  ( const avxf& a, const avxf& b, const avxf& c) { return a*b-c; }
  __forceinline const avxf nmadd ( const avxf& a, const avxf& b, const avxf& c) { return -a*b-c;}
  __forceinline const avxf nmsub ( const avxf& a, const avxf& b, const avxf& c) { return c-a*b; }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline avxf& operator +=( avxf& a, const avxf& b ) { return a = a + b; }
  __forceinline avxf& operator +=( avxf& a, const float b ) { return a = a + b; }

  __forceinline avxf& operator -=( avxf& a, const avxf& b ) { return a = a - b; }
  __forceinline avxf& operator -=( avxf& a, const float b ) { return a = a - b; }

  __forceinline avxf& operator *=( avxf& a, const avxf& b ) { return a = a * b; }
  __forceinline avxf& operator *=( avxf& a, const float b ) { return a = a * b; }

  __forceinline avxf& operator /=( avxf& a, const avxf& b ) { return a = a / b; }
  __forceinline avxf& operator /=( avxf& a, const float b ) { return a = a / b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const avxb operator ==( const avxf& a, const avxf& b ) { return _mm256_cmp_ps(a.m256, b.m256, _CMP_EQ_OQ ); }
  __forceinline const avxb operator ==( const avxf& a, const float b ) { return _mm256_cmp_ps(a.m256, avxf(b), _CMP_EQ_OQ ); }
  __forceinline const avxb operator ==( const float a, const avxf& b ) { return _mm256_cmp_ps(avxf(a), b.m256, _CMP_EQ_OQ ); }

  __forceinline const avxb operator !=( const avxf& a, const avxf& b ) { return _mm256_cmp_ps(a.m256, b.m256, _CMP_NEQ_OQ); }
  __forceinline const avxb operator !=( const avxf& a, const float b ) { return _mm256_cmp_ps(a.m256, avxf(b), _CMP_NEQ_OQ); }
  __forceinline const avxb operator !=( const float a, const avxf& b ) { return _mm256_cmp_ps(avxf(a), b.m256, _CMP_NEQ_OQ); }

  __forceinline const avxb operator < ( const avxf& a, const avxf& b ) { return _mm256_cmp_ps(a.m256, b.m256, _CMP_LT_OQ ); }
  __forceinline const avxb operator < ( const avxf& a, const float b ) { return _mm256_cmp_ps(a.m256, avxf(b), _CMP_LT_OQ ); }
  __forceinline const avxb operator < ( const float a, const avxf& b ) { return _mm256_cmp_ps(avxf(a), b.m256, _CMP_LT_OQ ); }

  __forceinline const avxb operator >=( const avxf& a, const avxf& b ) { return _mm256_cmp_ps(a.m256, b.m256, _CMP_GE_OQ); }
  __forceinline const avxb operator >=( const avxf& a, const float b ) { return _mm256_cmp_ps(a.m256, avxf(b), _CMP_GE_OQ); }
  __forceinline const avxb operator >=( const float a, const avxf& b ) { return _mm256_cmp_ps(avxf(a), b.m256, _CMP_GE_OQ); }

  __forceinline const avxb operator > ( const avxf& a, const avxf& b ) { return _mm256_cmp_ps(a.m256, b.m256, _CMP_GT_OQ); }
  __forceinline const avxb operator > ( const avxf& a, const float b ) { return _mm256_cmp_ps(a.m256, avxf(b), _CMP_GT_OQ); }
  __forceinline const avxb operator > ( const float a, const avxf& b ) { return _mm256_cmp_ps(avxf(a), b.m256, _CMP_GT_OQ); }

  __forceinline const avxb operator <=( const avxf& a, const avxf& b ) { return _mm256_cmp_ps(a.m256, b.m256, _CMP_LE_OQ ); }
  __forceinline const avxb operator <=( const avxf& a, const float b ) { return _mm256_cmp_ps(a.m256, avxf(b), _CMP_LE_OQ ); }
  __forceinline const avxb operator <=( const float a, const avxf& b ) { return _mm256_cmp_ps(avxf(a), b.m256, _CMP_LE_OQ ); }
  
  __forceinline const avxf select( const avxb& m, const avxf& t, const avxf& f ) { 
    return _mm256_blendv_ps(f, t, m); 
  }

#if defined(__clang__) || defined(_MSC_VER) && !defined(__INTEL_COMPILER)
  __forceinline const avxf select(const int m, const avxf& t, const avxf& f) {
	  return select(avxb(m), t, f); // workaround for clang and Microsoft compiler bugs
  }
#else
  __forceinline const avxf select( const int m, const avxf& t, const avxf& f ) { 
	  return _mm256_blend_ps(f, t, m);
  }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Rounding Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const avxf round_even( const avxf& a ) { return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT); }
  __forceinline const avxf round_down( const avxf& a ) { return _mm256_round_ps(a, _MM_FROUND_TO_NEG_INF    ); }
  __forceinline const avxf round_up  ( const avxf& a ) { return _mm256_round_ps(a, _MM_FROUND_TO_POS_INF    ); }
  __forceinline const avxf round_zero( const avxf& a ) { return _mm256_round_ps(a, _MM_FROUND_TO_ZERO       ); }
  __forceinline const avxf floor     ( const avxf& a ) { return _mm256_round_ps(a, _MM_FROUND_TO_NEG_INF    ); }
  __forceinline const avxf ceil      ( const avxf& a ) { return _mm256_round_ps(a, _MM_FROUND_TO_POS_INF    ); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline avxf unpacklo( const avxf& a, const avxf& b ) { return _mm256_unpacklo_ps(a.m256, b.m256); }
  __forceinline avxf unpackhi( const avxf& a, const avxf& b ) { return _mm256_unpackhi_ps(a.m256, b.m256); }

  template<size_t i> __forceinline const avxf shuffle( const avxf& a ) {
    return _mm256_permute_ps(a, _MM_SHUFFLE(i, i, i, i));
  }

  template<size_t i0, size_t i1> __forceinline const avxf shuffle( const avxf& a ) {
    return _mm256_permute2f128_ps(a, a, (i1 << 4) | (i0 << 0));
  }

  template<size_t i0, size_t i1> __forceinline const avxf shuffle( const avxf& a,  const avxf& b) {
    return _mm256_permute2f128_ps(a, b, (i1 << 4) | (i0 << 0));
  }

  template<size_t i0, size_t i1, size_t i2, size_t i3> __forceinline const avxf shuffle( const avxf& a ) {
    return _mm256_permute_ps(a, _MM_SHUFFLE(i3, i2, i1, i0));
  }

  template<size_t i0, size_t i1, size_t i2, size_t i3> __forceinline const avxf shuffle( const avxf& a, const avxf& b ) {
    return _mm256_shuffle_ps(a, b, _MM_SHUFFLE(i3, i2, i1, i0));
  }

  template<> __forceinline const avxf shuffle<0, 0, 2, 2>( const avxf& b ) { return _mm256_moveldup_ps(b); }
  template<> __forceinline const avxf shuffle<1, 1, 3, 3>( const avxf& b ) { return _mm256_movehdup_ps(b); }
  template<> __forceinline const avxf shuffle<0, 1, 0, 1>( const avxf& b ) { return _mm256_castpd_ps(_mm256_movedup_pd(_mm256_castps_pd(b))); }

  __forceinline const avxf broadcast(const float* ptr) { return _mm256_broadcast_ss(ptr); }
  template<size_t i> __forceinline const avxf insert (const avxf& a, const ssef& b) { return _mm256_insertf128_ps (a,b,i); }
  template<size_t i> __forceinline const ssef extract   (const avxf& a            ) { return _mm256_extractf128_ps(a  ,i); }
  template<>         __forceinline const ssef extract<0>(const avxf& a            ) { return _mm256_castps256_ps128(a); }

  template<size_t i> __forceinline float fextract   (const avxf& a            ) { return _mm_cvtss_f32(_mm256_extractf128_ps(a  ,i)); }

#if defined (__AVX2__)
  __forceinline avxf permute(const avxf &a, const __m256i &index) {
    return _mm256_permutevar8x32_ps(a,index);
  }

  template<int i>
  __forceinline avxf alignr(const avxf &a, const avxf &b) {
    return _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(a), _mm256_castps_si256(b), i));
  }  
#endif

#if defined (__AVX_I__)
  template<const int mode>
  __forceinline ssei convert_to_hf16(const avxf &a) {
    return _mm256_cvtps_ph(a,mode);
  }

  __forceinline avxf convert_from_hf16(const ssei &a) {
    return _mm256_cvtph_ps(a);
  }
#endif

  __forceinline ssef broadcast4f( const avxf& a, const size_t k ) {  
    return ssef::broadcast(&a[k]);
  }

  __forceinline avxf broadcast8f( const avxf& a, const size_t k ) {  
    return avxf::broadcast(&a[k]);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Transpose
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline void transpose(const avxf& r0, const avxf& r1, const avxf& r2, const avxf& r3, avxf& c0, avxf& c1, avxf& c2, avxf& c3)
  {
    avxf l02 = unpacklo(r0,r2);
    avxf h02 = unpackhi(r0,r2);
    avxf l13 = unpacklo(r1,r3);
    avxf h13 = unpackhi(r1,r3);
    c0 = unpacklo(l02,l13);
    c1 = unpackhi(l02,l13);
    c2 = unpacklo(h02,h13);
    c3 = unpackhi(h02,h13);
  }

  __forceinline void transpose(const avxf& r0, const avxf& r1, const avxf& r2, const avxf& r3, avxf& c0, avxf& c1, avxf& c2)
  {
    avxf l02 = unpacklo(r0,r2);
    avxf h02 = unpackhi(r0,r2);
    avxf l13 = unpacklo(r1,r3);
    avxf h13 = unpackhi(r1,r3);
    c0 = unpacklo(l02,l13);
    c1 = unpackhi(l02,l13);
    c2 = unpacklo(h02,h13);
  }

  __forceinline void transpose(const avxf& r0, const avxf& r1, const avxf& r2, const avxf& r3, const avxf& r4, const avxf& r5, const avxf& r6, const avxf& r7,
                               avxf& c0, avxf& c1, avxf& c2, avxf& c3, avxf& c4, avxf& c5, avxf& c6, avxf& c7)
  {
    avxf h0,h1,h2,h3; transpose(r0,r1,r2,r3,h0,h1,h2,h3);
    avxf h4,h5,h6,h7; transpose(r4,r5,r6,r7,h4,h5,h6,h7);
    c0 = shuffle<0,2>(h0,h4);
    c1 = shuffle<0,2>(h1,h5);
    c2 = shuffle<0,2>(h2,h6);
    c3 = shuffle<0,2>(h3,h7);
    c4 = shuffle<1,3>(h0,h4);
    c5 = shuffle<1,3>(h1,h5);
    c6 = shuffle<1,3>(h2,h6);
    c7 = shuffle<1,3>(h3,h7);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline const avxf vreduce_min2(const avxf& v) { return min(v,shuffle<1,0,3,2>(v)); }
  __forceinline const avxf vreduce_min4(const avxf& v) { avxf v1 = vreduce_min2(v); return min(v1,shuffle<2,3,0,1>(v1)); }
  __forceinline const avxf vreduce_min (const avxf& v) { avxf v1 = vreduce_min4(v); return min(v1,shuffle<1,0>(v1)); }

  __forceinline const avxf vreduce_max2(const avxf& v) { return max(v,shuffle<1,0,3,2>(v)); }
  __forceinline const avxf vreduce_max4(const avxf& v) { avxf v1 = vreduce_max2(v); return max(v1,shuffle<2,3,0,1>(v1)); }
  __forceinline const avxf vreduce_max (const avxf& v) { avxf v1 = vreduce_max4(v); return max(v1,shuffle<1,0>(v1)); }

  __forceinline const avxf vreduce_add2(const avxf& v) { return v + shuffle<1,0,3,2>(v); }
  __forceinline const avxf vreduce_add4(const avxf& v) { avxf v1 = vreduce_add2(v); return v1 + shuffle<2,3,0,1>(v1); }
  __forceinline const avxf vreduce_add (const avxf& v) { avxf v1 = vreduce_add4(v); return v1 + shuffle<1,0>(v1); }

  __forceinline float reduce_min(const avxf& v) { return _mm_cvtss_f32(extract<0>(vreduce_min(v))); }
  __forceinline float reduce_max(const avxf& v) { return _mm_cvtss_f32(extract<0>(vreduce_max(v))); }
  __forceinline float reduce_add(const avxf& v) { return _mm_cvtss_f32(extract<0>(vreduce_add(v))); }

  __forceinline size_t select_min(const avxf& v) { return __bsf(movemask(v == vreduce_min(v))); }
  __forceinline size_t select_max(const avxf& v) { return __bsf(movemask(v == vreduce_max(v))); }

  __forceinline size_t select_min(const avxb& valid, const avxf& v) { const avxf a = select(valid,v,avxf(pos_inf)); return __bsf(movemask(valid & (a == vreduce_min(a)))); }
  __forceinline size_t select_max(const avxb& valid, const avxf& v) { const avxf a = select(valid,v,avxf(neg_inf)); return __bsf(movemask(valid & (a == vreduce_max(a)))); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Memory load and store operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline avxf load8f( const void* const a) { 
    return _mm256_load_ps((const float*)a); 
  }

  __forceinline void store8f(void *ptr, const avxf& f ) { 
    return _mm256_store_ps((float*)ptr,f);
  }

  __forceinline void storeu8f(void *ptr, const avxf& f ) { 
    return _mm256_storeu_ps((float*)ptr,f);
  }

  __forceinline void store8f( const avxb& mask, void *ptr, const avxf& f ) { 
    return _mm256_maskstore_ps((float*)ptr,(__m256i)mask,f);
  }

#if defined (__AVX2__)
  __forceinline avxf load8f_nt(void* ptr) {
    return _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i*)ptr));
  }

#endif
  
  __forceinline void store8f_nt(void* ptr, const avxf& v) {
    _mm256_stream_ps((float*)ptr,v);
  }
 
  __forceinline const avxf broadcast4f(const void* ptr) { 
    return _mm256_broadcast_ps((__m128*)ptr); 
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Euclidian Space Operators
  ////////////////////////////////////////////////////////////////////////////////

  //__forceinline avxf dot ( const avxf& a, const avxf& b ) {
  //  return vreduce_add4(a*b);
  //}

  __forceinline avxf dot ( const avxf& a, const avxf& b ) {
    return _mm256_dp_ps(a,b,0x7F);
  }

  __forceinline avxf cross ( const avxf& a, const avxf& b ) 
  {
    const avxf a0 = a;
    const avxf b0 = shuffle<1,2,0,3>(b);
    const avxf a1 = shuffle<1,2,0,3>(a);
    const avxf b1 = b;
    return shuffle<1,2,0,3>(msub(a0,b0,a1*b1));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  inline std::ostream& operator<<(std::ostream& cout, const avxf& a) {
    return cout << "<" << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << ", " << a[4] << ", " << a[5] << ", " << a[6] << ", " << a[7] << ">";
  }
}

// bvh4_intersector_chunk.h
namespace embree
{
  namespace isa
  {
    /*! BVH4 packet traversal implementation. */
    template<int types, bool robust, typename PrimitiveIntersector>
      class BVH4Intersector8Chunk
    {
      /* shortcuts for frequently used types */
      typedef typename PrimitiveIntersector::Precalculations Precalculations;
      typedef typename PrimitiveIntersector::Primitive Primitive;
      typedef typename BVH4::NodeRef NodeRef;
      typedef typename BVH4::Node Node;
      static const size_t stackSize = 4*BVH4::maxDepth+1;
      
    public:
      static void intersect(avxb* valid, BVH4* bvh, Ray8& ray);
    };
  }
}

// bvh4_intersector_chunk.cpp
namespace embree
{
  namespace isa
  {
    template<int types, bool robust, typename PrimitiveIntersector8>
    void BVH4Intersector8Chunk<types, robust, PrimitiveIntersector8>::intersect(avxb* valid_i, BVH4* bvh, Ray8& ray)
    {
      /* load ray */
      const avxb valid0 = *valid_i;
      const avx3f rdir = rcp_safe(ray.dir);
      const avx3f org(ray.org), org_rdir = org * rdir;
      avxf ray_tnear = select(valid0,ray.tnear,pos_inf);
      avxf ray_tfar  = select(valid0,ray.tfar ,neg_inf);
      const avxf inf = avxf(pos_inf);
      Precalculations pre(valid0,ray);

      /* allocate stack and push root node */
      avxf    stack_near[stackSize];
      NodeRef stack_node[stackSize];
      stack_node[0] = BVH4::invalidNode;
      stack_near[0] = inf;
      stack_node[1] = bvh->root;
      stack_near[1] = ray_tnear; 
      NodeRef* stackEnd = stack_node+stackSize;
      NodeRef* __restrict__ sptr_node = stack_node + 2;
      avxf*    __restrict__ sptr_near = stack_near + 2;
      
      while (1)
      {
        /* pop next node from stack */
        assert(sptr_node > stack_node);
        sptr_node--;
        sptr_near--;
        NodeRef cur = *sptr_node;
        if (unlikely(cur == BVH4::invalidNode)) {
          assert(sptr_node == stack_node);
          break;
        }
        
        /* cull node if behind closest hit point */
        avxf curDist = *sptr_near;
        if (unlikely(none(ray_tfar > curDist))) 
          continue;
        
        while (1)
        {
          /* process normal nodes */
          if (likely((types & 0x1) && cur.isNode()))
          {
	    const avxb valid_node = ray_tfar > curDist;
	    STAT3(normal.trav_nodes,1,popcnt(valid_node),8);
	    const Node* __restrict__ const node = cur.node();
	    
	    /* pop of next node */
	    assert(sptr_node > stack_node);
	    sptr_node--;
	    sptr_near--;
	    cur = *sptr_node; 
	    curDist = *sptr_near;
	    
#pragma unroll(4)
	    for (unsigned i=0; i<BVH4::N; i++)
	    {
	      const NodeRef child = node->children[i];
	      if (unlikely(child == BVH4::emptyNode)) break;
	      avxf lnearP; const avxb lhit = node->intersect8<robust>(i,org,rdir,org_rdir,ray_tnear,ray_tfar,lnearP);
	      
	      /* if we hit the child we choose to continue with that child if it 
		 is closer than the current next child, or we push it onto the stack */
	      if (likely(any(lhit)))
	      {
		assert(sptr_node < stackEnd);
		assert(child != BVH4::emptyNode);
		const avxf childDist = select(lhit,lnearP,inf);
		sptr_node++;
		sptr_near++;
		
		/* push cur node onto stack and continue with hit child */
		if (any(childDist < curDist))
		{
		  *(sptr_node-1) = cur;
		  *(sptr_near-1) = curDist; 
		  curDist = childDist;
		  cur = child;
		}
		
		/* push hit child onto stack */
		else {
		  *(sptr_node-1) = child;
		  *(sptr_near-1) = childDist; 
		}
	      }     
	    }
	  }
	  
	  /* process motion blur nodes */
          else if (likely((types & 0x10) && cur.isNodeMB()))
	  {
	    const avxb valid_node = ray_tfar > curDist;
	    STAT3(normal.trav_nodes,1,popcnt(valid_node),8);
	    const BVH4::NodeMB* __restrict__ const node = cur.nodeMB();
          
	    /* pop of next node */
	    assert(sptr_node > stack_node);
	    sptr_node--;
	    sptr_near--;
	    cur = *sptr_node; 
	    curDist = *sptr_near;
	    
#pragma unroll(4)
	    for (unsigned i=0; i<BVH4::N; i++)
	    {
	      const NodeRef child = node->child(i);
	      if (unlikely(child == BVH4::emptyNode)) break;
	      avxf lnearP; const avxb lhit = node->intersect(i,org,rdir,org_rdir,ray_tnear,ray_tfar,ray.time,lnearP);
	      	      
	      /* if we hit the child we choose to continue with that child if it 
		 is closer than the current next child, or we push it onto the stack */
	      if (likely(any(lhit)))
	      {
		assert(sptr_node < stackEnd);
		assert(child != BVH4::emptyNode);
		const avxf childDist = select(lhit,lnearP,inf);
		sptr_node++;
		sptr_near++;
		
		/* push cur node onto stack and continue with hit child */
		if (any(childDist < curDist))
		{
		  *(sptr_node-1) = cur;
		  *(sptr_near-1) = curDist; 
		  curDist = childDist;
		  cur = child;
		}
		
		/* push hit child onto stack */
		else {
		  *(sptr_node-1) = child;
		  *(sptr_near-1) = childDist; 
		}
	      }	      
	    }
	  }
	  else 
	    break;
	}
        
	/* return if stack is empty */
	if (unlikely(cur == BVH4::invalidNode)) {
	  assert(sptr_node == stack_node);
	  break;
	}
	
	/* intersect leaf */
	assert(cur != BVH4::emptyNode);
	const avxb valid_leaf = ray_tfar > curDist;
	STAT3(normal.trav_leaves,1,popcnt(valid_leaf),8);
	size_t items; const Primitive* prim = (Primitive*) cur.leaf(items);
	PrimitiveIntersector8::intersect(valid_leaf,pre,ray,prim,items,bvh->scene);
	ray_tfar = select(valid_leaf,ray.tfar,ray_tfar);
      }
      AVX_ZERO_UPPER();
    }
    
    // BVH4Intersector8Chunk::occluded removed

    DEFINE_INTERSECTOR8(BVH4Bezier1vIntersector8Chunk, BVH4Intersector8Chunk<0x1 COMMA false COMMA LeafIterator8<Bezier1vIntersector8<LeafMode> > >);
    DEFINE_INTERSECTOR8(BVH4Bezier1iIntersector8Chunk, BVH4Intersector8Chunk<0x1 COMMA false COMMA LeafIterator8<Bezier1iIntersector8<LeafMode> > >);
    DEFINE_INTERSECTOR8(BVH4Triangle1Intersector8ChunkMoeller, BVH4Intersector8Chunk<0x1 COMMA false COMMA LeafIterator8<Triangle1Intersector8MoellerTrumbore<LeafMode> > >);
    DEFINE_INTERSECTOR8(BVH4Triangle4Intersector8ChunkMoeller, BVH4Intersector8Chunk<0x1 COMMA false COMMA LeafIterator8<Triangle4Intersector8MoellerTrumbore<LeafMode COMMA true> > >);
    DEFINE_INTERSECTOR8(BVH4Triangle4Intersector8ChunkMoellerNoFilter, BVH4Intersector8Chunk<0x1 COMMA false COMMA LeafIterator8<Triangle4Intersector8MoellerTrumbore<LeafMode COMMA false> > >);
    DEFINE_INTERSECTOR8(BVH4Triangle8Intersector8ChunkMoeller, BVH4Intersector8Chunk<0x1 COMMA false COMMA LeafIterator8<Triangle8Intersector8MoellerTrumbore<LeafMode COMMA true> > >);
    DEFINE_INTERSECTOR8(BVH4Triangle8Intersector8ChunkMoellerNoFilter, BVH4Intersector8Chunk<0x1 COMMA false COMMA LeafIterator8<Triangle8Intersector8MoellerTrumbore<LeafMode COMMA false> > >);
    DEFINE_INTERSECTOR8(BVH4Triangle1vIntersector8ChunkPluecker, BVH4Intersector8Chunk<0x1 COMMA true COMMA LeafIterator8<Triangle1vIntersector8Pluecker<LeafMode> > >);
    DEFINE_INTERSECTOR8(BVH4Triangle4vIntersector8ChunkPluecker, BVH4Intersector8Chunk<0x1 COMMA true COMMA LeafIterator8<Triangle4vIntersector8Pluecker<LeafMode> > >);
    DEFINE_INTERSECTOR8(BVH4Triangle4iIntersector8ChunkPluecker, BVH4Intersector8Chunk<0x1 COMMA true COMMA LeafIterator8<Triangle4iIntersector8Pluecker<LeafMode> > >);
    DEFINE_INTERSECTOR8(BVH4VirtualIntersector8Chunk, BVH4Intersector8Chunk<0x1 COMMA false COMMA LeafIterator8<VirtualAccelIntersector8> >);

    DEFINE_INTERSECTOR8(BVH4Triangle1vMBIntersector8ChunkMoeller, BVH4Intersector8Chunk<0x10 COMMA false COMMA LeafIterator8<Triangle1vIntersector8MoellerTrumboreMB<LeafMode> > >);
    DEFINE_INTERSECTOR8(BVH4Triangle4vMBIntersector8ChunkMoeller, BVH4Intersector8Chunk<0x10 COMMA false COMMA LeafIterator8<Triangle4vMBIntersector8MoellerTrumbore<LeafMode COMMA true> > >);
  }
}

// triangle4_intersector8_moeller.h
namespace embree
{
  namespace isa
  {
    /*! Intersector for 4 triangles with 8 rays. This intersector
     *  implements a modified version of the Moeller Trumbore
     *  intersector from the paper "Fast, Minimum Storage Ray-Triangle
     *  Intersection". In contrast to the paper we precalculate some
     *  factors and factor the calculations differently to allow
     *  precalculating the cross product e1 x e2. */
    template<bool list, bool enableIntersectionFilter>
      struct Triangle4Intersector8MoellerTrumbore
      {
        typedef Triangle4 Primitive;
        
        struct Precalculations {
          __forceinline Precalculations (const avxb& valid, const Ray8& ray) {}
        };
        
        /*! Intersects a 8 rays with 4 triangles. */
        static __forceinline void intersect(const avxb& valid_i, Precalculations& pre, Ray8& ray, const Primitive& tri, Scene* scene)
        {
          for (size_t i=0; i<4; i++)
          {
            if (!tri.valid(i)) break;
            STAT3(normal.trav_prims,1,popcnt(valid_i),8);
            
            /* load edges and geometry normal */
            avxb valid = valid_i;
            const avx3f p0 = broadcast8f(tri.v0,i);
            const avx3f e1 = broadcast8f(tri.e1,i);
            const avx3f e2 = broadcast8f(tri.e2,i);
            const avx3f Ng = broadcast8f(tri.Ng,i);
            
            /* calculate denominator */
            const avx3f C = p0 - ray.org;
            const avx3f R = cross(ray.dir,C);
            const avxf den = dot(Ng,ray.dir);
            const avxf absDen = abs(den);
            const avxf sgnDen = signmsk(den);
            
            /* test against edge p2 p0 */
            const avxf U = dot(R,e2) ^ sgnDen;
            valid &= U >= 0.0f;
            
            /* test against edge p0 p1 */
            const avxf V = dot(R,e1) ^ sgnDen;
            valid &= V >= 0.0f;
            
            /* test against edge p1 p2 */
            const avxf W = absDen-U-V;
            valid &= W >= 0.0f;
            if (likely(none(valid))) continue;
            
            /* perform depth test */
            const avxf T = dot(Ng,C) ^ sgnDen;
            valid &= (T >= absDen*ray.tnear) & (absDen*ray.tfar >= T);
            if (unlikely(none(valid))) continue;
            
            /* perform backface culling */
#if defined(RTCORE_BACKFACE_CULLING)
            valid &= den > avxf(zero);
            if (unlikely(none(valid))) continue;
#else
            valid &= den != avxf(zero);
            if (unlikely(none(valid))) continue;
#endif
            
            /* ray masking test */
#if defined(RTCORE_RAY_MASK)
            valid &= (tri.mask[i] & ray.mask) != 0;
            if (unlikely(none(valid))) continue;
#endif
            
            /* calculate hit information */
            const avxf rcpAbsDen = rcp(absDen);
            const avxf u = U*rcpAbsDen;
            const avxf v = V*rcpAbsDen;
            const avxf t = T*rcpAbsDen;
            const int geomID = tri.geomID<list>(i);
            const int primID = tri.primID<list>(i);
            
            /* intersection filter test */
#if defined(RTCORE_INTERSECTION_FILTER)
            if (enableIntersectionFilter) {
              Geometry* geometry = scene->get(geomID);
              if (unlikely(geometry->hasIntersectionFilter8())) {
                runIntersectionFilter8(valid,geometry,ray,u,v,t,Ng,geomID,primID);
                continue;
              }
            }
#endif
            
            /* update hit information */
            store8f(valid,&ray.u,u);
            store8f(valid,&ray.v,v);
            store8f(valid,&ray.tfar,t);
            store8i(valid,&ray.geomID,geomID);
            store8i(valid,&ray.primID,primID);
            store8f(valid,&ray.Ng.x,Ng.x);
            store8f(valid,&ray.Ng.y,Ng.y);
            store8f(valid,&ray.Ng.z,Ng.z);
          }
        }
        
        // Triangle4Intersector8MoellerTrumbore::occluded removed
      };
  }
}

// bvh4.h
#include "embree2/rtcore.h"
#include "common/alloc.h"
#include "common/accel.h"
#include "common/scene.h"
#include "geometry/primitive.h"
#include "common/ray.h"

namespace embree
{
  /*! Multi BVH with 4 children. Each node stores the bounding box of
   * it's 4 children as well as 4 child pointers. */
  class BVH4 : public AccelData
  {
    ALIGNED_CLASS;
  public:
    
    /*! forward declaration of node type */
    struct BaseNode;
    struct Node;
    struct NodeMB;
    struct UnalignedNode;
    struct NodeSingleSpaceMB;
    struct NodeDualSpaceMB;
    struct NodeConeMB;
#define BVH4HAIR_MB_VERSION 0

#if BVH4HAIR_MB_VERSION == 0
    typedef NodeSingleSpaceMB UnalignedNodeMB;
#elif BVH4HAIR_MB_VERSION == 1
    typedef NodeDualSpaceMB UnalignedNodeMB;
#elif BVH4HAIR_MB_VERSION == 2
    typedef NodeConeMB UnalignedNodeMB;
#endif

    /*! branching width of the tree */
    static const size_t N = 4;

    /*! Number of address bits the Node and primitives are aligned
        to. Maximally 2^alignment-1 many primitive blocks per leaf are
        supported. */
    static const size_t alignment = 4;

    /*! highest address bit is used as barrier for some algorithms */
    static const size_t barrier_mask = (1LL << (8*sizeof(size_t)-1));

    /*! Masks the bits that store the number of items per leaf. */
    static const size_t align_mask = (1 << alignment)-1;  
    static const size_t items_mask = (1 << alignment)-1;  

    /*! different supported node types */
    static const size_t tyNode = 0;
    static const size_t tyNodeMB = 1;
    static const size_t tyUnalignedNode = 2;
    static const size_t tyUnalignedNodeMB = 3;
    static const size_t tyLeaf = 8;

    /*! Empty node */
    static const size_t emptyNode = tyLeaf;

    /*! Invalid node, used as marker in traversal */
    static const size_t invalidNode = (((size_t)-1) & (~items_mask)) | tyLeaf;
      
    /*! Maximal depth of the BVH. */
    static const size_t maxBuildDepth = 32;
    static const size_t maxBuildDepthLeaf = maxBuildDepth+16;
    static const size_t maxDepth = maxBuildDepthLeaf+maxBuildDepthLeaf+maxBuildDepth;
    
    /*! Maximal number of primitive blocks in a leaf. */
    static const size_t maxLeafBlocks = items_mask-tyLeaf;

    /*! Cost of one traversal step. */
    static const int travCost = 1;
    static const int travCostAligned = 2;
    static const int travCostUnaligned = 3; // FIXME: find best cost
    static const int intCost = 6;

    /*! Pointer that points to a node or a list of primitives */
    struct NodeRef
    {
      /*! Default constructor */
      __forceinline NodeRef () {}

      /*! Construction from integer */
      __forceinline NodeRef (size_t ptr) : ptr(ptr) {}

      /*! Cast to size_t */
      __forceinline operator size_t() const { return ptr; }

      /*! Prefetches the node this reference points to */
      __forceinline void prefetch(int types) const {
	prefetchL1(((char*)ptr)+0*64);
	prefetchL1(((char*)ptr)+1*64);
	if (types > 0x1) {
	  prefetchL1(((char*)ptr)+2*64);
	  prefetchL1(((char*)ptr)+3*64);
	  /*prefetchL1(((char*)ptr)+4*64);
	  prefetchL1(((char*)ptr)+5*64);
	  prefetchL1(((char*)ptr)+6*64);
	  prefetchL1(((char*)ptr)+7*64);*/
	}
      }

      /*! Sets the barrier bit. */
      __forceinline void setBarrier() { ptr |= barrier_mask; }
      
      /*! Clears the barrier bit. */
      __forceinline void clearBarrier() { ptr &= ~barrier_mask; }

      /*! Checks if this is an barrier. A barrier tells the top level tree rotations how deep to enter the tree. */
      __forceinline bool isBarrier() const { return (ptr & barrier_mask) != 0; }

      /*! checks if this is a leaf */
      __forceinline size_t isLeaf() const { return ptr & tyLeaf; }

      /*! checks if this is a leaf */
      __forceinline int isLeaf(int types) const { 
	if      (types == 0x0001) return !isNode();
	/*else if (types == 0x0010) return !isNodeMB();
	else if (types == 0x0100) return !isUnalignedNode();
	else if (types == 0x1000) return !isUnalignedNodeMB();*/
	else return isLeaf();
      }
      
      /*! checks if this is a node */
      __forceinline int isNode() const { return (ptr & (size_t)align_mask) == tyNode; }
      __forceinline int isNode(int types) const { return (types == 0x1) || ((types & 0x1) && isNode()); }

      // MB nodes removed

      /*! returns base node pointer */
      __forceinline BaseNode* baseNode(int types) { 
	assert(!isLeaf()); 
	if (types == 0x1) return (BaseNode*)ptr; 
	else              return (BaseNode*)(ptr & ~(size_t)align_mask); 
      }
      __forceinline const BaseNode* baseNode(int types) const { 
	assert(!isLeaf()); 
	if (types == 0x1) return (const BaseNode*)ptr; 
	else              return (const BaseNode*)(ptr & ~(size_t)align_mask); 
      }

      /*! returns node pointer */
      __forceinline       Node* node()       { assert(isNode()); return (      Node*)ptr; }
      __forceinline const Node* node() const { assert(isNode()); return (const Node*)ptr; }

      // MB nodes removed

      /*! returns leaf pointer */
      __forceinline char* leaf(size_t& num) const {
        assert(isLeaf());
        num = (ptr & (size_t)items_mask)-tyLeaf;
        return (char*)(ptr & ~(size_t)align_mask);
      }

      /*! clear all bit flags */
      __forceinline void clearFlags() {
        ptr &= ~(size_t)align_mask;
      }

    private:
      size_t ptr;
    };
    
    /*! BVH4 Base Node */
    struct BaseNode
    {
      /*! Clears the node. */
      __forceinline void clear() {
	for (size_t i=0; i<N; i++) children[i] = emptyNode;
      }

        /*! Returns reference to specified child */
      __forceinline       NodeRef& child(size_t i)       { assert(i<N); return children[i]; }
      __forceinline const NodeRef& child(size_t i) const { assert(i<N); return children[i]; }

      /*! verifies the node */
      __forceinline bool verify() const  // FIXME: call in statistics
      {
	for (size_t i=0; i<BVH4::N; i++) {
	  if (child(i) == BVH4::emptyNode) {
	    for (; i<BVH4::N; i++) {
	      if (child(i) != BVH4::emptyNode)
		return false;
	    }
	    break;
	  }
	}
	return true;
      }

      NodeRef children[N];    //!< Pointer to the 4 children (can be a node or leaf)
    };
    
    /*! BVH4 Node */
    struct Node : public BaseNode
    {
      /*! Clears the node. */
      __forceinline void clear() {
        lower_x = lower_y = lower_z = pos_inf; 
        upper_x = upper_y = upper_z = neg_inf;
	BaseNode::clear();
      }

      /*! Sets bounding box and ID of child. */
      __forceinline void set(size_t i, const NodeRef& childID) {
	assert(i < N);
        children[i] = childID;
      }

      /*! Sets bounding box of child. */
      __forceinline void set(size_t i, const BBox3fa& bounds) 
      {
        assert(i < N);
        lower_x[i] = bounds.lower.x; lower_y[i] = bounds.lower.y; lower_z[i] = bounds.lower.z;
        upper_x[i] = bounds.upper.x; upper_y[i] = bounds.upper.y; upper_z[i] = bounds.upper.z;
      }

      /*! Sets bounding box and ID of child. */
      __forceinline void set(size_t i, const BBox3fa& bounds, const NodeRef& childID) {
        set(i,bounds);
        children[i] = childID;
      }

      /*! Returns bounds of node. */
      __forceinline BBox3fa bounds() const {
        const Vec3fa lower(reduce_min(lower_x),reduce_min(lower_y),reduce_min(lower_z));
        const Vec3fa upper(reduce_max(upper_x),reduce_max(upper_y),reduce_max(upper_z));
        return BBox3fa(lower,upper);
      }

      /*! Returns bounds of specified child. */
      __forceinline BBox3fa bounds(size_t i) const 
      {
        assert(i < N);
        const Vec3fa lower(lower_x[i],lower_y[i],lower_z[i]);
        const Vec3fa upper(upper_x[i],upper_y[i],upper_z[i]);
        return BBox3fa(lower,upper);
      }

      /*! Returns extent of bounds of specified child. */
      __forceinline BBox3fa extend(size_t i) const {
	return bounds(i).size();
      }

      /*! Returns bounds of all children */
      __forceinline void bounds(BBox<ssef>& bounds0, BBox<ssef>& bounds1, BBox<ssef>& bounds2, BBox<ssef>& bounds3) const {
        transpose(lower_x,lower_y,lower_z,ssef(zero),bounds0.lower,bounds1.lower,bounds2.lower,bounds3.lower);
        transpose(upper_x,upper_y,upper_z,ssef(zero),bounds0.upper,bounds1.upper,bounds2.upper,bounds3.upper);
      }

      /*! swap two children of the node */
      __forceinline void swap(size_t i, size_t j)
      {
	assert(i<N && j<N);
	std::swap(children[i],children[j]);
	std::swap(lower_x[i],lower_x[j]);
	std::swap(lower_y[i],lower_y[j]);
	std::swap(lower_z[i],lower_z[j]);
	std::swap(upper_x[i],upper_x[j]);
	std::swap(upper_y[i],upper_y[j]);
	std::swap(upper_z[i],upper_z[j]);
      }

      /*! Returns reference to specified child */
      __forceinline       NodeRef& child(size_t i)       { assert(i<N); return children[i]; }
      __forceinline const NodeRef& child(size_t i) const { assert(i<N); return children[i]; }

     // single/packet4 intersections removed
      
      /*! intersection with ray packet of size 8 */
#if defined(__AVX__)
      template<bool robust>
      __forceinline avxb intersect8(size_t i, const avx3f& org, const avx3f& rdir, const avx3f& org_rdir, const avxf& tnear, const avxf& tfar, avxf& dist) const
      {
#if defined(__AVX2__)
	const avxf lclipMinX = msub(lower_x[i],rdir.x,org_rdir.x);
	const avxf lclipMinY = msub(lower_y[i],rdir.y,org_rdir.y);
	const avxf lclipMinZ = msub(lower_z[i],rdir.z,org_rdir.z);
	const avxf lclipMaxX = msub(upper_x[i],rdir.x,org_rdir.x);
	const avxf lclipMaxY = msub(upper_y[i],rdir.y,org_rdir.y);
	const avxf lclipMaxZ = msub(upper_z[i],rdir.z,org_rdir.z);
#else
	const avxf lclipMinX = (lower_x[i] - org.x) * rdir.x;
	const avxf lclipMinY = (lower_y[i] - org.y) * rdir.y;
	const avxf lclipMinZ = (lower_z[i] - org.z) * rdir.z;
	const avxf lclipMaxX = (upper_x[i] - org.x) * rdir.x;
	const avxf lclipMaxY = (upper_y[i] - org.y) * rdir.y;
	const avxf lclipMaxZ = (upper_z[i] - org.z) * rdir.z;
#endif

        if (robust) {
          const float round_down = 1.0f-2.0f*float(ulp);
          const float round_up   = 1.0f+2.0f*float(ulp);
          const avxf lnearP = max(max(min(lclipMinX, lclipMaxX), min(lclipMinY, lclipMaxY)), min(lclipMinZ, lclipMaxZ));
          const avxf lfarP  = min(min(max(lclipMinX, lclipMaxX), max(lclipMinY, lclipMaxY)), max(lclipMinZ, lclipMaxZ));
          const avxb lhit   = round_down*max(lnearP,tnear) <= round_up*min(lfarP,tfar);      
          dist = lnearP;
          return lhit;
        }

#if defined(__AVX2__)
	const avxf lnearP = maxi(maxi(mini(lclipMinX, lclipMaxX), mini(lclipMinY, lclipMaxY)), mini(lclipMinZ, lclipMaxZ));
	const avxf lfarP  = mini(mini(maxi(lclipMinX, lclipMaxX), maxi(lclipMinY, lclipMaxY)), maxi(lclipMinZ, lclipMaxZ));
	const avxb lhit   = maxi(lnearP,tnear) <= mini(lfarP,tfar);      
#else
	const avxf lnearP = max(max(min(lclipMinX, lclipMaxX), min(lclipMinY, lclipMaxY)), min(lclipMinZ, lclipMaxZ));
	const avxf lfarP  = min(min(max(lclipMinX, lclipMaxX), max(lclipMinY, lclipMaxY)), max(lclipMinZ, lclipMaxZ));
	const avxb lhit   = max(lnearP,tnear) <= min(lfarP,tfar);      
#endif
	dist = lnearP;
	return lhit;
      }
#endif
      
    public:
      ssef lower_x;           //!< X dimension of lower bounds of all 4 children.
      ssef upper_x;           //!< X dimension of upper bounds of all 4 children.
      ssef lower_y;           //!< Y dimension of lower bounds of all 4 children.
      ssef upper_y;           //!< Y dimension of upper bounds of all 4 children.
      ssef lower_z;           //!< Z dimension of lower bounds of all 4 children.
      ssef upper_z;           //!< Z dimension of upper bounds of all 4 children.
    };

    // MB nodes removed

  public:

    /*! BVH4 default constructor. */
    BVH4 (const PrimitiveType& primTy, Scene* scene, bool listMode);

    /*! BVH4 destruction */
    ~BVH4 ();

    // BVH instanciations removed

  public:
    const PrimitiveType& primTy;       //!< primitive type stored in the BVH
    Scene* scene;                      //!< scene pointer
    bool listMode;                     //!< true if number of leaf items not encoded in NodeRef
    NodeRef root;                      //!< Root node
    size_t numPrimitives;
    size_t numVertices;

    /*! data arrays for fast builders */
  public:
    std::vector<BVH4*> objects;
  };

}
