#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define SC_GENERATED 1
#define USING_GENERIC_ENTRY false
#define CONCURRENT 1

#include <algorithm>
#include <vector>
#include <unordered_set>
//#include <valgrind/callgrind.h>
#include <iomanip>
#include <fstream>
#include <locale>

#include "sc/mmap.hpp"
#include "sc/TransactionManager.h"

#include "sc/ExecutionProfiler.h"

using namespace std;

#ifdef NUMWARE
  const int numWare = NUMWARE;
#else
  const int numWare = 2;
#endif
#ifdef NUMPROG
  const size_t numPrograms = NUMPROG;
#else
  const size_t numPrograms = 100;
#endif

uint failedOS = 0;
uint failedDel = 0;
uint failedNO = 0;

const size_t warehouseTblSize = 8 * (numWare / 8 + 1);
const size_t itemTblSize = 100000 * 1.5;
const size_t districtTblSize = 8 * ((numWare * 10) / 8 + 1);
const size_t customerTblSize = districtTblSize * 3000;
const size_t orderTblSize = customerTblSize * 1.5 + 0.5 * numPrograms;
const size_t newOrderTblSize = orderTblSize * 0.3 + 0.5 * numPrograms;
const size_t orderLineTblSize = orderTblSize * 12;
const size_t stockTblSize = numWare * itemTblSize;
const size_t historyTblSize = orderTblSize;

const size_t districtTblArrayLengths[] = {1};
const size_t customerTblArrayLengths[] = {1,1};
const size_t itemTblArrayLengths[] = {1};
const size_t historyTblArrayLengths[] = {1};
const size_t orderTblArrayLengths[] = {1,1};
const size_t newOrderTblArrayLengths[] = {1,1};
const size_t warehouseTblArrayLengths[] = {1};
const size_t stockTblArrayLengths[] = {1};
const size_t orderLineTblArrayLengths[] = {1,1};

const size_t warehouseTblPoolSizes[] = {8, 0};
const size_t itemTblPoolSizes[] = {65536*2, 0};
const size_t districtTblPoolSizes[] = {16, 0};
const size_t customerTblPoolSizes[] = {16384*2, 0, 16384};
const size_t orderTblPoolSizes[] = {262144*2, 65536, 0};
const size_t newOrderTblPoolSizes[] = {8192*2, 2048, 0};
const size_t orderLineTblPoolSizes[] = {4194304*2, 1048576, 2097152};
const size_t stockTblPoolSizes[] = {65536*2, 0};
const size_t historyTblPoolSizes[] = {262144*2, 65536};
     

struct SEntry5_IISDS {
  int _1;  int _2;  PString _3;  double _4;  PString _5;  bool isInvalid;
  SEntry5_IISDS() :_1(-2147483648), _2(-2147483648), _3(), _4(-1.7976931348623157E308), _5(), isInvalid(false){}
  SEntry5_IISDS(const int& _1, const int& _2, const PString& _3, const double& _4, const PString& _5) : _1(_1), _2(_2), _3(_3), _4(_4), _5(_5), isInvalid(false){}
  FORCE_INLINE SEntry5_IISDS* copy() const {  SEntry5_IISDS* ptr = (SEntry5_IISDS*) malloc(sizeof(SEntry5_IISDS)); new(ptr) SEntry5_IISDS(_1, _2, _3, _4, _5);  return ptr;}
};
struct SEntry11_IISSSSSSDDI {
  int _1;  int _2;  PString _3;  PString _4;  PString _5;  PString _6;  PString _7;  PString _8;  double _9;  double _10;  int _11;  bool isInvalid;
  SEntry11_IISSSSSSDDI() :_1(-2147483648), _2(-2147483648), _3(), _4(), _5(), _6(), _7(), _8(), _9(-1.7976931348623157E308), _10(-1.7976931348623157E308), _11(-2147483648), isInvalid(false){}
  SEntry11_IISSSSSSDDI(const int& _1, const int& _2, const PString& _3, const PString& _4, const PString& _5, const PString& _6, const PString& _7, const PString& _8, const double& _9, const double& _10, const int& _11) : _1(_1), _2(_2), _3(_3), _4(_4), _5(_5), _6(_6), _7(_7), _8(_8), _9(_9), _10(_10), _11(_11), isInvalid(false){}
  FORCE_INLINE SEntry11_IISSSSSSDDI* copy() const {  SEntry11_IISSSSSSDDI* ptr = (SEntry11_IISSSSSSDDI*) malloc(sizeof(SEntry11_IISSSSSSDDI)); new(ptr) SEntry11_IISSSSSSDDI(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11);  return ptr;}
};
struct SEntry3_III {
  int _1;  int _2;  int _3;  bool isInvalid;
  SEntry3_III() :_1(-2147483648), _2(-2147483648), _3(-2147483648), isInvalid(false){}
  SEntry3_III(const int& _1, const int& _2, const int& _3) : _1(_1), _2(_2), _3(_3), isInvalid(false){}
  FORCE_INLINE SEntry3_III* copy() const {  SEntry3_III* ptr = (SEntry3_III*) malloc(sizeof(SEntry3_III)); new(ptr) SEntry3_III(_1, _2, _3);  return ptr;}
};
struct SEntry21_IIISSSSSSSSSTSDDDDIIS {
  int _1;  int _2;  int _3;  PString _4;  PString _5;  PString _6;  PString _7;  PString _8;  PString _9;  PString _10;  PString _11;  PString _12;  date _13;  PString _14;  double _15;  double _16;  double _17;  double _18;  int _19;  int _20;  PString _21;  bool isInvalid;
  SEntry21_IIISSSSSSSSSTSDDDDIIS() :_1(-2147483648), _2(-2147483648), _3(-2147483648), _4(), _5(), _6(), _7(), _8(), _9(), _10(), _11(), _12(), _13(0), _14(), _15(-1.7976931348623157E308), _16(-1.7976931348623157E308), _17(-1.7976931348623157E308), _18(-1.7976931348623157E308), _19(-2147483648), _20(-2147483648), _21(), isInvalid(false){}
  SEntry21_IIISSSSSSSSSTSDDDDIIS(const int& _1, const int& _2, const int& _3, const PString& _4, const PString& _5, const PString& _6, const PString& _7, const PString& _8, const PString& _9, const PString& _10, const PString& _11, const PString& _12, const date& _13, const PString& _14, const double& _15, const double& _16, const double& _17, const double& _18, const int& _19, const int& _20, const PString& _21) : _1(_1), _2(_2), _3(_3), _4(_4), _5(_5), _6(_6), _7(_7), _8(_8), _9(_9), _10(_10), _11(_11), _12(_12), _13(_13), _14(_14), _15(_15), _16(_16), _17(_17), _18(_18), _19(_19), _20(_20), _21(_21), isInvalid(false){}
  FORCE_INLINE SEntry21_IIISSSSSSSSSTSDDDDIIS* copy() const {  SEntry21_IIISSSSSSSSSTSDDDDIIS* ptr = (SEntry21_IIISSSSSSSSSTSDDDDIIS*) malloc(sizeof(SEntry21_IIISSSSSSSSSTSDDDDIIS)); new(ptr) SEntry21_IIISSSSSSSSSTSDDDDIIS(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21);  return ptr;}
};
struct SEntry8_IIIITIIB {
  int _1;  int _2;  int _3;  int _4;  date _5;  int _6;  int _7;  int _8;  bool isInvalid;
  SEntry8_IIIITIIB() :_1(-2147483648), _2(-2147483648), _3(-2147483648), _4(-2147483648), _5(0), _6(-2147483648), _7(-2147483648), _8(0), isInvalid(false){}
  SEntry8_IIIITIIB(const int& _1, const int& _2, const int& _3, const int& _4, const date& _5, const int& _6, const int& _7, const int& _8) : _1(_1), _2(_2), _3(_3), _4(_4), _5(_5), _6(_6), _7(_7), _8(_8), isInvalid(false){}
  FORCE_INLINE SEntry8_IIIITIIB* copy() const {  SEntry8_IIIITIIB* ptr = (SEntry8_IIIITIIB*) malloc(sizeof(SEntry8_IIIITIIB)); new(ptr) SEntry8_IIIITIIB(_1, _2, _3, _4, _5, _6, _7, _8);  return ptr;}
};
struct SEntry8_IIIIITDS {
  int _1;  int _2;  int _3;  int _4;  int _5;  date _6;  double _7;  PString _8;  bool isInvalid;
  SEntry8_IIIIITDS() :_1(-2147483648), _2(-2147483648), _3(-2147483648), _4(-2147483648), _5(-2147483648), _6(0), _7(-1.7976931348623157E308), _8(), isInvalid(false){}
  SEntry8_IIIIITDS(const int& _1, const int& _2, const int& _3, const int& _4, const int& _5, const date& _6, const double& _7, const PString& _8) : _1(_1), _2(_2), _3(_3), _4(_4), _5(_5), _6(_6), _7(_7), _8(_8), isInvalid(false){}
  FORCE_INLINE SEntry8_IIIIITDS* copy() const {  SEntry8_IIIIITDS* ptr = (SEntry8_IIIIITDS*) malloc(sizeof(SEntry8_IIIIITDS)); new(ptr) SEntry8_IIIIITDS(_1, _2, _3, _4, _5, _6, _7, _8);  return ptr;}
};
struct SEntry17_IIISSSSSSSSSSIIIS {
  int _1;  int _2;  int _3;  PString _4;  PString _5;  PString _6;  PString _7;  PString _8;  PString _9;  PString _10;  PString _11;  PString _12;  PString _13;  int _14;  int _15;  int _16;  PString _17;  bool isInvalid;
  SEntry17_IIISSSSSSSSSSIIIS() :_1(-2147483648), _2(-2147483648), _3(-2147483648), _4(), _5(), _6(), _7(), _8(), _9(), _10(), _11(), _12(), _13(), _14(-2147483648), _15(-2147483648), _16(-2147483648), _17(), isInvalid(false){}
  SEntry17_IIISSSSSSSSSSIIIS(const int& _1, const int& _2, const int& _3, const PString& _4, const PString& _5, const PString& _6, const PString& _7, const PString& _8, const PString& _9, const PString& _10, const PString& _11, const PString& _12, const PString& _13, const int& _14, const int& _15, const int& _16, const PString& _17) : _1(_1), _2(_2), _3(_3), _4(_4), _5(_5), _6(_6), _7(_7), _8(_8), _9(_9), _10(_10), _11(_11), _12(_12), _13(_13), _14(_14), _15(_15), _16(_16), _17(_17), isInvalid(false){}
  FORCE_INLINE SEntry17_IIISSSSSSSSSSIIIS* copy() const {  SEntry17_IIISSSSSSSSSSIIIS* ptr = (SEntry17_IIISSSSSSSSSSIIIS*) malloc(sizeof(SEntry17_IIISSSSSSSSSSIIIS)); new(ptr) SEntry17_IIISSSSSSSSSSIIIS(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17);  return ptr;}
};
struct SEntry10_IIIIIITIDS {
  int _1;  int _2;  int _3;  int _4;  int _5;  int _6;  date _7;  int _8;  double _9;  PString _10;  bool isInvalid;
  SEntry10_IIIIIITIDS() :_1(-2147483648), _2(-2147483648), _3(-2147483648), _4(-2147483648), _5(-2147483648), _6(-2147483648), _7(0), _8(-2147483648), _9(-1.7976931348623157E308), _10(), isInvalid(false){}
  SEntry10_IIIIIITIDS(const int& _1, const int& _2, const int& _3, const int& _4, const int& _5, const int& _6, const date& _7, const int& _8, const double& _9, const PString& _10) : _1(_1), _2(_2), _3(_3), _4(_4), _5(_5), _6(_6), _7(_7), _8(_8), _9(_9), _10(_10), isInvalid(false){}
  FORCE_INLINE SEntry10_IIIIIITIDS* copy() const {  SEntry10_IIIIIITIDS* ptr = (SEntry10_IIIIIITIDS*) malloc(sizeof(SEntry10_IIIIIITIDS)); new(ptr) SEntry10_IIIIIITIDS(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10);  return ptr;}
};
struct SEntry9_ISSSSSSDD {
  int _1;  PString _2;  PString _3;  PString _4;  PString _5;  PString _6;  PString _7;  double _8;  double _9;  bool isInvalid;
  SEntry9_ISSSSSSDD() :_1(-2147483648), _2(), _3(), _4(), _5(), _6(), _7(), _8(-1.7976931348623157E308), _9(-1.7976931348623157E308), isInvalid(false){}
  SEntry9_ISSSSSSDD(const int& _1, const PString& _2, const PString& _3, const PString& _4, const PString& _5, const PString& _6, const PString& _7, const double& _8, const double& _9) : _1(_1), _2(_2), _3(_3), _4(_4), _5(_5), _6(_6), _7(_7), _8(_8), _9(_9), isInvalid(false){}
  FORCE_INLINE SEntry9_ISSSSSSDD* copy() const {  SEntry9_ISSSSSSDD* ptr = (SEntry9_ISSSSSSDD*) malloc(sizeof(SEntry9_ISSSSSSDD)); new(ptr) SEntry9_ISSSSSSDD(_1, _2, _3, _4, _5, _6, _7, _8, _9);  return ptr;}
};
bool operator== (const SEntry5_IISDS& o1, const SEntry5_IISDS& o2) {
  if (o1.isInvalid || o2.isInvalid) return o1.isInvalid && o2.isInvalid;
  else return o1._1 == o2._1 && 
  o1._2 == o2._2 && 
  o1._3 == o2._3 && 
  (fabs(o1._4 - o2._4) < 0.01) && 
  o1._5 == o2._5;
}
bool operator== (const SEntry11_IISSSSSSDDI& o1, const SEntry11_IISSSSSSDDI& o2) {
  if (o1.isInvalid || o2.isInvalid) return o1.isInvalid && o2.isInvalid;
  else return o1._1 == o2._1 && 
  o1._2 == o2._2 && 
  o1._3 == o2._3 && 
  o1._4 == o2._4 && 
  o1._5 == o2._5 && 
  o1._6 == o2._6 && 
  o1._7 == o2._7 && 
  o1._8 == o2._8 && 
  (fabs(o1._9 - o2._9) < 0.01) && 
  (fabs(o1._10 - o2._10) < 0.01) && 
  o1._11 == o2._11;
}
bool operator== (const SEntry3_III& o1, const SEntry3_III& o2) {
  if (o1.isInvalid || o2.isInvalid) return o1.isInvalid && o2.isInvalid;
  else return o1._1 == o2._1 && 
  o1._2 == o2._2 && 
  o1._3 == o2._3;
}
bool operator== (const SEntry21_IIISSSSSSSSSTSDDDDIIS& o1, const SEntry21_IIISSSSSSSSSTSDDDDIIS& o2) {
  if (o1.isInvalid || o2.isInvalid) return o1.isInvalid && o2.isInvalid;
  else return o1._1 == o2._1 && 
  o1._2 == o2._2 && 
  o1._3 == o2._3 && 
  o1._4 == o2._4 && 
  o1._5 == o2._5 && 
  o1._6 == o2._6 && 
  o1._7 == o2._7 && 
  o1._8 == o2._8 && 
  o1._9 == o2._9 && 
  o1._10 == o2._10 && 
  o1._11 == o2._11 && 
  o1._12 == o2._12 && 
  o1._13 == o2._13 && 
  o1._14 == o2._14 && 
  (fabs(o1._15 - o2._15) < 0.01) && 
  (fabs(o1._16 - o2._16) < 0.01) && 
  (fabs(o1._17 - o2._17) < 0.01) && 
  (fabs(o1._18 - o2._18) < 0.01) && 
  o1._19 == o2._19 && 
  o1._20 == o2._20 && 
  o1._21 == o2._21;
}
bool operator== (const SEntry8_IIIITIIB& o1, const SEntry8_IIIITIIB& o2) {
  if (o1.isInvalid || o2.isInvalid) return o1.isInvalid && o2.isInvalid;
  else return o1._1 == o2._1 && 
  o1._2 == o2._2 && 
  o1._3 == o2._3 && 
  o1._4 == o2._4 && 
  o1._5 == o2._5 && 
  o1._6 == o2._6 && 
  o1._7 == o2._7 && 
  o1._8 == o2._8;
}
bool operator== (const SEntry8_IIIIITDS& o1, const SEntry8_IIIIITDS& o2) {
  if (o1.isInvalid || o2.isInvalid) return o1.isInvalid && o2.isInvalid;
  else return o1._1 == o2._1 && 
  o1._2 == o2._2 && 
  o1._3 == o2._3 && 
  o1._4 == o2._4 && 
  o1._5 == o2._5 && 
  o1._6 == o2._6 && 
  (fabs(o1._7 - o2._7) < 0.01) && 
  o1._8 == o2._8;
}
bool operator== (const SEntry17_IIISSSSSSSSSSIIIS& o1, const SEntry17_IIISSSSSSSSSSIIIS& o2) {
  if (o1.isInvalid || o2.isInvalid) return o1.isInvalid && o2.isInvalid;
  else return o1._1 == o2._1 && 
  o1._2 == o2._2 && 
  o1._3 == o2._3 && 
  o1._4 == o2._4 && 
  o1._5 == o2._5 && 
  o1._6 == o2._6 && 
  o1._7 == o2._7 && 
  o1._8 == o2._8 && 
  o1._9 == o2._9 && 
  o1._10 == o2._10 && 
  o1._11 == o2._11 && 
  o1._12 == o2._12 && 
  o1._13 == o2._13 && 
  o1._14 == o2._14 && 
  o1._15 == o2._15 && 
  o1._16 == o2._16 && 
  o1._17 == o2._17;
}
bool operator== (const SEntry10_IIIIIITIDS& o1, const SEntry10_IIIIIITIDS& o2) {
  if (o1.isInvalid || o2.isInvalid) return o1.isInvalid && o2.isInvalid;
  else return o1._1 == o2._1 && 
  o1._2 == o2._2 && 
  o1._3 == o2._3 && 
  o1._4 == o2._4 && 
  o1._5 == o2._5 && 
  o1._6 == o2._6 && 
  o1._7 == o2._7 && 
  o1._8 == o2._8 && 
  (fabs(o1._9 - o2._9) < 0.01) && 
  o1._10 == o2._10;
}
bool operator== (const SEntry9_ISSSSSSDD& o1, const SEntry9_ISSSSSSDD& o2) {
  if (o1.isInvalid || o2.isInvalid) return o1.isInvalid && o2.isInvalid;
  else return o1._1 == o2._1 && 
  o1._2 == o2._2 && 
  o1._3 == o2._3 && 
  o1._4 == o2._4 && 
  o1._5 == o2._5 && 
  o1._6 == o2._6 && 
  o1._7 == o2._7 && 
  (fabs(o1._8 - o2._8) < 0.01) && 
  (fabs(o1._9 - o2._9) < 0.01);
}
#if USING_GENERIC_ENTRY
struct GenericOps_3214 {  //OL 0
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    int x1 = e.get(1).data.i;
    int x2 = (x1 << 2) + e.get(3).data.i;
    int x3 = (x2 << 4) + e.get(2).data.i;
    int x4 = (x3 << 4) + e.get(4).data.i;
    return x4;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) {
    if(e1.get(3) != e2.get(3) || e1.get(2) != e2.get(2) || e1.get(1) != e2.get(1) || e1.get(4) != e2.get(4))
       return 1;
    return 0;
  }
};
struct GenericOps_23 { //NO 1
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    int x1 = e.get(3).data.i;
    int x2 = (x1 << 4) + e.get(2).data.i;
    return x2;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) {
    if(e1.get(2) != e2.get(2) || e1.get(3) != e2.get(3))
       return 1;
    return 0;
  }
};
struct GenericOps_321 { //C0  O0   NO0
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    int x1 = e.get(1).data.i;
    int x2 = (x1 << 2) + e.get(3).data.i;
    int x3 = (x2 << 4) + e.get(2).data.i;
    return x3;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) {
    if(e1.get(3) != e2.get(3) || e1.get(2) != e2.get(2) || e1.get(1) != e2.get(1))
       return 1;
    return 0;
  }
};
struct GenericOps_236 { //C 1
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    int x1 = HASH(e.get(6).data.s);
    int x2 = (x1 << 2) + e.get(3).data.i;
    int x3 = (x2 << 4) + e.get(2).data.i;
    return x3;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) {
    if(e1.get(2) != e2.get(2) || e1.get(3) != e2.get(3) || e1.get(6) != e2.get(6))
       return 1;
    return 0;
  }
};
struct GenericOps_123 { //OL1
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    int x1 = e.get(1).data.i;
    int x2 = (x1 << 2) + e.get(3).data.i;
    int x3 = (x2 << 4) + e.get(2).data.i;
    return x3;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) {
    if(e1.get(1) != e2.get(1) || e1.get(2) != e2.get(2) || e1.get(3) != e2.get(3))
       return 1;
    return 0;
  }
};
struct GenericOps_12345678 { // H0
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    unsigned int h = 0;
    h = h ^ (HASH(e.get(1)) + 0x9e3779b9 + (h<<6) + (h>>2));
    h = h ^ (HASH(e.get(2)) + 0x9e3779b9 + (h<<6) + (h>>2));
    h = h ^ (HASH(e.get(3)) + 0x9e3779b9 + (h<<6) + (h>>2));
    h = h ^ (HASH(e.get(4)) + 0x9e3779b9 + (h<<6) + (h>>2));
    h = h ^ (HASH(e.get(5)) + 0x9e3779b9 + (h<<6) + (h>>2));
    h = h ^ (HASH(e.get(6)) + 0x9e3779b9 + (h<<6) + (h>>2));
    h = h ^ (HASH(e.get(7)) + 0x9e3779b9 + (h<<6) + (h>>2));
    h = h ^ (HASH(e.get(8)) + 0x9e3779b9 + (h<<6) + (h>>2));
    return h;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) {
    if(e1.get(1) != e2.get(1) || e1.get(2) != e2.get(2) || e1.get(3) != e2.get(3) || e1.get(4) != e2.get(4) || e1.get(5) != e2.get(5) || e1.get(6) != e2.get(6) || e1.get(7) != e2.get(7) || e1.get(8) != e2.get(8))
       return 1;
    return 0;
  }
};
struct GenericOps_234 {  //O 1
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    int x1 = e.get(4).data.i;
    int x2 = (x1 << 2) + e.get(3).data.i;
    int x3 = (x2 << 4) + e.get(2).data.i;
    return x3;

  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) {
    if(e1.get(2) != e2.get(2) || e1.get(3) != e2.get(3) || e1.get(4) != e2.get(4))
       return 1;
    return 0;
  }
};
struct GenericCmp_236_4 {
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    int x1 = HASH(e.get(6));
    int x2 = (x1 << 2) + e.get(3).data.i;
    int x3 = (x2 << 4) + e.get(2).data.i;
    return x3;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) {
    const Any &r1 = e1.get(4);
    const Any &r2 = e2.get(4);
    if (r1 == r2)
      return 0;
    else if( r1 < r2)
      return -1;
    else
      return 1;

  }
};
struct GenericCmp_23_1 {
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    int x1 = e.getInt(3);
    int x2 = (x1 << 4) + e.getInt(2);
    return x2;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) {
    const Any &r1 = e1.get(1);
    const Any &r2 = e2.get(1);
    if (r1 == r2)
      return 0;
    else if( r1 < r2)
      return -1;
    else
      return 1;

  }
};
struct GenericCmp_234_1 {
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    int x1 = e.get(4).data.i;
    int x2 = (x1 << 2) + e.get(3).data.i;
    int x3 = (x2 << 4) + e.get(2).data.i;
    return x3;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) {
    const Any &r1 = e1.get(1);
    const Any &r2 = e2.get(1);
    if (r1 == r2)
      return 0;
    else if( r1 < r2)
      return -1;
    else
      return 1;

  }
};
struct GenericFixedRange_1f1t100002 {
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    return e.getInt(1)-1;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) { return 0;}
};
struct GenericFixedRange_3f1t6_2f1t11_1f1t3001 {
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    return ((e.getInt(3)-1) * 10 + e.getInt(2)-1)*3000 + e.getInt(1)-1;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) { return 0;}
};
struct GenericFixedRange_2f1t6_1f1t100001 {
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    return (e.getInt(2)-1)* 100000 + e.getInt(1)-1;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) { return 0;}
};
struct GenericFixedRange_1f1t6 {
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    return  e.getInt(1) - 1;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) { return 0;}
};
struct GenericFixedRange_2f1t6_1f1t11 {
  FORCE_INLINE static size_t hash(const GenericEntry& e) {
    return (e.getInt(2)-1) * 10 + e.getInt(1)-1;;
  }
  FORCE_INLINE static char cmp(const GenericEntry& e1, const GenericEntry& e2) { return 0;}
};
#else
struct SEntry8_IIIITIIB_Idx234 {  // O 1
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry8_IIIITIIB& x3468)  {
    int x1 = x3468._4;
    int x2 = (x1 << 2) + x3468._3;
    int x3 = (x2 << 4) + x3468._2;
    return x3;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry8_IIIITIIB& x3502, const struct SEntry8_IIIITIIB& x3503) {
    int x3504 = 0;
    if(((x3502._2)==((x3503._2)))) {
      if(((x3502._3)==((x3503._3)))) {
        if(((x3502._4)==((x3503._4)))) {
          x3504 = 0;
        } else {
          x3504 = 1;
        };
      } else {
        x3504 = 1;
      };
    } else {
      x3504 = 1;
    };
    int x3521 = x3504;
    return x3521;
  }
};
 struct SEntry10_IIIIIITIDS_Idx3214 { // OL 0
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry10_IIIIIITIDS& x3576)  {
    int x1 = x3576._1;
    int x2 = (x1 << 2) + x3576._3;
    int x3 = (x2 << 4) + x3576._2;
    int x4 = (x3 << 4) + x3576._4;
    return x4;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry10_IIIIIITIDS& x3620, const struct SEntry10_IIIIIITIDS& x3621) {
    int x3622 = 0;
    if(((x3620._3)==((x3621._3)))) {
      if(((x3620._2)==((x3621._2)))) {
        if(((x3620._1)==((x3621._1)))) {
          if(((x3620._4)==((x3621._4)))) {
            x3622 = 0;
          } else {
            x3622 = 1;
          };
        } else {
          x3622 = 1;
        };
      } else {
        x3622 = 1;
      };
    } else {
      x3622 = 1;
    };
    int x3644 = x3622;
    return x3644;
  }
};
 struct SEntry21_IIISSSSSSSSSTSDDDDIIS_Idx321 {  //C 0
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3709)  {
    int x2 = x3709._3;
    int x3 = (x2 << 4) + x3709._2;
    int x1 = (x3 << 12) + x3709._1;
    return x1;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3743, const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3744) {
    int x3745 = 0;
    if(((x3743._3)==((x3744._3)))) {
      if(((x3743._2)==((x3744._2)))) {
        if(((x3743._1)==((x3744._1)))) {
          x3745 = 0;
        } else {
          x3745 = 1;
        };
      } else {
        x3745 = 1;
      };
    } else {
      x3745 = 1;
    };
    int x3762 = x3745;
    return x3762;
  }
};
 struct SEntry8_IIIIITDS_Idx12345678 {  //H Idx0
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry8_IIIIITDS& x3215)  {
    int x3216 = 0;
    int x3217 = x3216;
    x3216 = (x3217^(((((HASH((x3215._1)))+(-1640531527))+((x3217<<(6))))+((x3217>>(2))))));
    int x3227 = x3216;
    x3216 = (x3227^(((((HASH((x3215._2)))+(-1640531527))+((x3227<<(6))))+((x3227>>(2))))));
    int x3237 = x3216;
    x3216 = (x3237^(((((HASH((x3215._3)))+(-1640531527))+((x3237<<(6))))+((x3237>>(2))))));
    int x3247 = x3216;
    x3216 = (x3247^(((((HASH((x3215._4)))+(-1640531527))+((x3247<<(6))))+((x3247>>(2))))));
    int x3257 = x3216;
    x3216 = (x3257^(((((HASH((x3215._5)))+(-1640531527))+((x3257<<(6))))+((x3257>>(2))))));
    int x3267 = x3216;
    x3216 = (x3267^(((((HASH((x3215._6)))+(-1640531527))+((x3267<<(6))))+((x3267>>(2))))));
    int x3277 = x3216;
    x3216 = (x3277^(((((HASH((x3215._7)))+(-1640531527))+((x3277<<(6))))+((x3277>>(2))))));
    int x3287 = x3216;
    x3216 = (x3287^(((((HASH((x3215._8)))+(-1640531527))+((x3287<<(6))))+((x3287>>(2))))));
    int x3297 = x3216;
    return x3297;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry8_IIIIITDS& x3299, const struct SEntry8_IIIIITDS& x3300) {
    int x3301 = 0;
    if(((x3299._1)==((x3300._1)))) {
      if(((x3299._2)==((x3300._2)))) {
        if(((x3299._3)==((x3300._3)))) {
          if(((x3299._4)==((x3300._4)))) {
            if(((x3299._5)==((x3300._5)))) {
              if(((x3299._6)==((x3300._6)))) {
                if(((x3299._7)==((x3300._7)))) {
                  if(((x3299._8)==((x3300._8)))) {
                    x3301 = 0;
                  } else {
                    x3301 = 1;
                  };
                } else {
                  x3301 = 1;
                };
              } else {
                x3301 = 1;
              };
            } else {
              x3301 = 1;
            };
          } else {
            x3301 = 1;
          };
        } else {
          x3301 = 1;
        };
      } else {
        x3301 = 1;
      };
    } else {
      x3301 = 1;
    };
    int x3343 = x3301;
    return x3343;
  }
};
 struct SEntry8_IIIITIIB_Idx321 {  // O 0
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry8_IIIITIIB& x3412)  {
    int x1 = x3412._1;
    int x2 = (x1 << 2) + x3412._3;
    int x3 = (x2 << 4) + x3412._2;
    return x3;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry8_IIIITIIB& x3446, const struct SEntry8_IIIITIIB& x3447) {
    int x3448 = 0;
    if(((x3446._3)==((x3447._3)))) {
      if(((x3446._2)==((x3447._2)))) {
        if(((x3446._1)==((x3447._1)))) {
          x3448 = 0;
        } else {
          x3448 = 1;
        };
      } else {
        x3448 = 1;
      };
    } else {
      x3448 = 1;
    };
    int x3465 = x3448;
    return x3465;
  }
};
 struct SEntry17_IIISSSSSSSSSSIIIS_Idx21 { // S 0
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry17_IIISSSSSSSSSSIIIS& x3826)  {
    int x1 = x3826._1;
    int x2 = (x1 << 2) + x3826._2;
    return x2;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry17_IIISSSSSSSSSSIIIS& x3850, const struct SEntry17_IIISSSSSSSSSSIIIS& x3851) {
    int x3852 = 0;
    if(((x3850._2)==((x3851._2)))) {
      if(((x3850._1)==((x3851._1)))) {
        x3852 = 0;
      } else {
        x3852 = 1;
      };
    } else {
      x3852 = 1;
    };
    int x3864 = x3852;
    return x3864;
  }
};
 struct SEntry3_III_Idx23 {  //NO Idx 1
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry3_III& x3168)  {
    int x1 = x3168._3;
    int x2 = (x1 << 4) + x3168._2;
    return x2;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry3_III& x3192, const struct SEntry3_III& x3193) {
    int x3194 = 0;
    if(((x3192._2)==((x3193._2)))) {
      if(((x3192._3)==((x3193._3)))) {
        x3194 = 0;
      } else {
        x3194 = 1;
      };
    } else {
      x3194 = 1;
    };
    int x3206 = x3194;
    return x3206;
  }
};
 struct SEntry10_IIIIIITIDS_Idx123 {  // OL 1
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry10_IIIIIITIDS& x3647)  {
    int x1 = x3647._1;
    int x2 = (x1 << 2) + x3647._3;
    int x3 = (x2 << 4) + x3647._2;
    return x3;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry10_IIIIIITIDS& x3681, const struct SEntry10_IIIIIITIDS& x3682) {
    int x3683 = 0;
    if(((x3681._1)==((x3682._1)))) {
      if(((x3681._2)==((x3682._2)))) {
        if(((x3681._3)==((x3682._3)))) {
          x3683 = 0;
        } else {
          x3683 = 1;
        };
      } else {
        x3683 = 1;
      };
    } else {
      x3683 = 1;
    };
    int x3700 = x3683;
    return x3700;
  }
};
 struct SEntry3_III_Idx321 {  //NO Idx0
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry3_III& x3112)  {
    int x1 = x3112._1;
    int x2 = (x1 << 2) + x3112._3;
    int x3 = (x2 << 4) + x3112._2;
    return x3;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry3_III& x3146, const struct SEntry3_III& x3147) {
    int x3148 = 0;
    if(((x3146._3)==((x3147._3)))) {
      if(((x3146._2)==((x3147._2)))) {
        if(((x3146._1)==((x3147._1)))) {
          x3148 = 0;
        } else {
          x3148 = 1;
        };
      } else {
        x3148 = 1;
      };
    } else {
      x3148 = 1;
    };
    int x3165 = x3148;
    return x3165;
  }
};
 struct SEntry5_IISDS_Idx1 { // I 0
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry5_IISDS& x3381)  {
    int x3382 = 0;
    int x3383 = x3382;
    x3382 = (x3383^(((((HASH((x3381._1)))+(-1640531527))+((x3383<<(6))))+((x3383>>(2))))));
    int x3393 = x3382;
    return x3393;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry5_IISDS& x3395, const struct SEntry5_IISDS& x3396) {
    int x3397 = 0;
    if(((x3395._1)==((x3396._1)))) {
      x3397 = 0;
    } else {
      x3397 = 1;
    };
    int x3404 = x3397;
    return x3404;
  }
};
 struct SEntry21_IIISSSSSSSSSTSDDDDIIS_Idx236 { // C 1
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3765)  {
    int x1 = HASH(x3765._6);
    int x2 = (x1 << 2) + x3765._3;
    int x3 = (x2 << 4) + x3765._2;
    return x3;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3799, const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3800) {
    int x3801 = 0;
    if(((x3799._2)==((x3800._2)))) {
      if(((x3799._3)==((x3800._3)))) {
        if(((x3799._6)==((x3800._6)))) {
          x3801 = 0;
        } else {
          x3801 = 1;
        };
      } else {
        x3801 = 1;
      };
    } else {
      x3801 = 1;
    };
    int x3818 = x3801;
    return x3818;
  }
};
 struct SEntry9_ISSSSSSDD_Idx1 { // W 0
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry9_ISSSSSSDD& x3351)  {
    int x3352 = 0;
    int x3353 = x3352;
    x3352 = (x3353^(((((HASH((x3351._1)))+(-1640531527))+((x3353<<(6))))+((x3353>>(2))))));
    int x3363 = x3352;
    return x3363;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry9_ISSSSSSDD& x3365, const struct SEntry9_ISSSSSSDD& x3366) {
    int x3367 = 0;
    if(((x3365._1)==((x3366._1)))) {
      x3367 = 0;
    } else {
      x3367 = 1;
    };
    int x3374 = x3367;
    return x3374;
  }
};
 struct SEntry11_IISSSSSSDDI_Idx21 { // D 0
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry11_IISSSSSSDDI& x3530)  {
    int x3531 = 0;
    int x3532 = x3531;
    x3531 = (x3532^(((((HASH((x3530._2)))+(-1640531527))+((x3532<<(6))))+((x3532>>(2))))));
    int x3542 = x3531;
    x3531 = (x3542^(((((HASH((x3530._1)))+(-1640531527))+((x3542<<(6))))+((x3542>>(2))))));
    int x3552 = x3531;
    return x3552;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry11_IISSSSSSDDI& x3554, const struct SEntry11_IISSSSSSDDI& x3555) {
    int x3556 = 0;
    if(((x3554._2)==((x3555._2)))) {
      if(((x3554._1)==((x3555._1)))) {
        x3556 = 0;
      } else {
        x3556 = 1;
      };
    } else {
      x3556 = 1;
    };
    int x3568 = x3556;
    return x3568;
  }
};

struct SEntry9_ISSSSSSDD_Idx1f1t6 {
#define int unsigned int

  FORCE_INLINE static size_t hash(const struct SEntry9_ISSSSSSDD& x3379) {
    return (x3379._1)-1;
  }
#undef int

  FORCE_INLINE static char cmp(const struct SEntry9_ISSSSSSDD& x3376, const struct SEntry9_ISSSSSSDD& x3377) {
    return 0;
  }
};

struct SEntry11_IISSSSSSDDI_Idx2f1t6_1f1t11 {
#define int unsigned int

  FORCE_INLINE static size_t hash(const struct SEntry11_IISSSSSSDDI& x3535) {
    return (x3535._2-1) * 10 + x3535._1-1;
  }
#undef int

  FORCE_INLINE static char cmp(const struct SEntry11_IISSSSSSDDI& x3532, const struct SEntry11_IISSSSSSDDI& x3533) {
    return 0;
  }
};

struct SEntry5_IISDS_Idx1f1t100002 {
#define int unsigned int

  FORCE_INLINE static size_t hash(const struct SEntry5_IISDS& x3398) {
    return (x3398._1)-1;
  }
#undef int

  FORCE_INLINE static char cmp(const struct SEntry5_IISDS& x3395, const struct SEntry5_IISDS& x3396) {
    return 0;
  }
};

struct SEntry21_IIISSSSSSSSSTSDDDDIIS_Idx3f1t6_2f1t11_1f1t3001 {
#define int unsigned int

  FORCE_INLINE static size_t hash(const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3693) {
    return ((x3693._3-1) * 10 + x3693._2-1)*3000 + x3693._1-1;
  }
#undef int

  FORCE_INLINE static char cmp(const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3690, const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3691) {
    return 0;
  }
};

struct SEntry17_IIISSSSSSSSSSIIIS_Idx2f1t6_1f1t100001 {
#define int unsigned int

  FORCE_INLINE static size_t hash(const struct SEntry17_IIISSSSSSSSSSIIIS& x3781) {
    return (x3781._2-1)* 100000 + x3781._1-1;
  }
#undef int

  FORCE_INLINE static char cmp(const struct SEntry17_IIISSSSSSSSSSIIIS& x3778, const struct SEntry17_IIISSSSSSSSSSIIIS& x3779) {
    return 0;
  }
};

struct SEntry3_III_Idx23_1 {
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry3_III& x3168)  {
    int x1 = x3168._3;
    int x2 = (x1 << 4) + x3168._2;
    return x2;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry3_III& x3251, const struct SEntry3_III& x3252) {
    int x3253 = x3251._1;
    int x3254 = x3252._1;
    return ((x3253==(x3254)) ? 0 : ((x3253>(x3254)) ? 1 : -1));
  }
};
 struct SEntry21_IIISSSSSSSSSTSDDDDIIS_Idx236_4 {
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3765)  {
    int x1 = HASH(x3765._6);
    int x2 = (x1 << 2) + x3765._3;
    int x3 = (x2 << 4) + x3765._2;
    return x3;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3944, const struct SEntry21_IIISSSSSSSSSTSDDDDIIS& x3945) {
    int x3948 = strcmpi((x3944._4).data_, (x3945._4).data_);
    return ((x3948>(0)) ? 1 : ((x3948<(0)) ? -1 : 0));
  }
};
 struct SEntry8_IIIITIIB_Idx234_1 {
  #define int unsigned int
  FORCE_INLINE static size_t hash(const struct SEntry8_IIIITIIB& x3468)  {
    int x1 = x3468._4;
    int x2 = (x1 << 2) + x3468._3;
    int x3 = (x2 << 4) + x3468._2;
    return x3;
  }
  #undef int
  FORCE_INLINE static char cmp(const struct SEntry8_IIIITIIB& x3599, const struct SEntry8_IIIITIIB& x3600) {
    int x3601 = x3599._1;
    int x3602 = x3600._1;
    return ((x3601==(x3602)) ? 0 : ((x3601>(x3602)) ? 1 : -1));
  }
};
#endif

      
typedef CuckooIndex<struct SEntry3_III, SEntry3_III_Idx321> newOrderTblIdx0Type;
typedef ConcurrentCuckooSecondaryIndex<struct SEntry3_III, SEntry3_III_Idx23> newOrderTblIdx1Type;
typedef MultiHashMapMV<struct SEntry3_III,newOrderTblIdx0Type, newOrderTblIdx1Type> newOrderTblStoreType;

typedef CuckooIndex<struct SEntry8_IIIIITDS, SEntry8_IIIIITDS_Idx12345678> historyTblIdx0Type;
typedef MultiHashMapMV<struct SEntry8_IIIIITDS,historyTblIdx0Type> historyTblStoreType;

typedef CuckooIndex<struct SEntry9_ISSSSSSDD, SEntry9_ISSSSSSDD_Idx1> warehouseTblIdx0Type;
typedef MultiHashMapMV<struct SEntry9_ISSSSSSDD,warehouseTblIdx0Type> warehouseTblStoreType;

typedef CuckooIndex<struct SEntry5_IISDS, SEntry5_IISDS_Idx1> itemTblIdx0Type;
typedef MultiHashMapMV<struct SEntry5_IISDS,itemTblIdx0Type> itemTblStoreType;

typedef CuckooIndex<struct SEntry8_IIIITIIB, SEntry8_IIIITIIB_Idx321> orderTblIdx0Type;
typedef ConcurrentCuckooSecondaryIndex<struct SEntry8_IIIITIIB, SEntry8_IIIITIIB_Idx234> orderTblIdx1Type;
typedef MultiHashMapMV<struct SEntry8_IIIITIIB,orderTblIdx0Type, orderTblIdx1Type> orderTblStoreType;

typedef CuckooIndex<struct SEntry11_IISSSSSSDDI, SEntry11_IISSSSSSDDI_Idx21> districtTblIdx0Type;
typedef MultiHashMapMV<struct SEntry11_IISSSSSSDDI,districtTblIdx0Type> districtTblStoreType;

typedef CuckooIndex<struct SEntry10_IIIIIITIDS, SEntry10_IIIIIITIDS_Idx3214> orderLineTblIdx0Type;
typedef ConcurrentCuckooSecondaryIndex<struct SEntry10_IIIIIITIDS, SEntry10_IIIIIITIDS_Idx123> orderLineTblIdx1Type;
typedef MultiHashMapMV<struct SEntry10_IIIIIITIDS,orderLineTblIdx0Type, orderLineTblIdx1Type> orderLineTblStoreType;

typedef CuckooIndex<struct SEntry21_IIISSSSSSSSSTSDDDDIIS, SEntry21_IIISSSSSSSSSTSDDDDIIS_Idx321> customerTblIdx0Type;
typedef ConcurrentCuckooSecondaryIndex<struct SEntry21_IIISSSSSSSSSTSDDDDIIS, SEntry21_IIISSSSSSSSSTSDDDDIIS_Idx236> customerTblIdx1Type;
typedef MultiHashMapMV<struct SEntry21_IIISSSSSSSSSTSDDDDIIS,customerTblIdx0Type, customerTblIdx1Type> customerTblStoreType;

typedef CuckooIndex<struct SEntry17_IIISSSSSSSSSSIIIS, SEntry17_IIISSSSSSSSSSIIIS_Idx21> stockTblIdx0Type;
typedef MultiHashMapMV<struct SEntry17_IIISSSSSSSSSSIIIS,stockTblIdx0Type> stockTblStoreType;

struct TPCC_Data {
  TPCC_Data(): 
  newOrderTbl(), newOrderTblIdx0(*(newOrderTblIdx0Type *)newOrderTbl.index[0]), newOrderTblIdx1(*(newOrderTblIdx1Type *)newOrderTbl.index[1]), 
  historyTbl(), historyTblIdx0(*(historyTblIdx0Type *)historyTbl.index[0]), 
  warehouseTbl(), warehouseTblIdx0(*(warehouseTblIdx0Type *)warehouseTbl.index[0]), 
  itemTbl(), itemTblIdx0(*(itemTblIdx0Type *)itemTbl.index[0]), 
  orderTbl(), orderTblIdx0(*(orderTblIdx0Type *)orderTbl.index[0]), orderTblIdx1(*(orderTblIdx1Type *)orderTbl.index[1]), 
  districtTbl(), districtTblIdx0(*(districtTblIdx0Type *)districtTbl.index[0]), 
  orderLineTbl(), orderLineTblIdx0(*(orderLineTblIdx0Type *)orderLineTbl.index[0]), orderLineTblIdx1(*(orderLineTblIdx1Type *)orderLineTbl.index[1]), 
  customerTbl(), customerTblIdx0(*(customerTblIdx0Type *)customerTbl.index[0]), customerTblIdx1(*(customerTblIdx1Type *)customerTbl.index[1]), 
  stockTbl(), stockTblIdx0(*(stockTblIdx0Type *)stockTbl.index[0]){}
  
  newOrderTblStoreType  newOrderTbl;  newOrderTblIdx0Type& newOrderTblIdx0;  newOrderTblIdx1Type& newOrderTblIdx1;
  historyTblStoreType  historyTbl;  historyTblIdx0Type& historyTblIdx0;
  warehouseTblStoreType  warehouseTbl;  warehouseTblIdx0Type& warehouseTblIdx0;
  itemTblStoreType  itemTbl;  itemTblIdx0Type& itemTblIdx0;
  orderTblStoreType  orderTbl;  orderTblIdx0Type& orderTblIdx0;  orderTblIdx1Type& orderTblIdx1;
  districtTblStoreType  districtTbl;  districtTblIdx0Type& districtTblIdx0;
  orderLineTblStoreType  orderLineTbl;  orderLineTblIdx0Type& orderLineTblIdx0;  orderLineTblIdx1Type& orderLineTblIdx1;
  customerTblStoreType  customerTbl;  customerTblIdx0Type& customerTblIdx0;  customerTblIdx1Type& customerTblIdx1;
  stockTblStoreType  stockTbl;  stockTblIdx0Type& stockTblIdx0;
};
struct ALIGN ThreadLocal { 
  
  uint8_t threadId;
  uint threadXactCounts[5];
  
  ThreadLocal(uint8_t tid, TPCC_Data& t): threadId(tid), 
  newOrderTbl(t.newOrderTbl), newOrderTblIdx0(t.newOrderTblIdx0), newOrderTblIdx1(t.newOrderTblIdx1), 
  historyTbl(t.historyTbl), historyTblIdx0(t.historyTblIdx0), 
  warehouseTbl(t.warehouseTbl), warehouseTblIdx0(t.warehouseTblIdx0), 
  itemTbl(t.itemTbl), itemTblIdx0(t.itemTblIdx0), 
  orderTbl(t.orderTbl), orderTblIdx0(t.orderTblIdx0), orderTblIdx1(t.orderTblIdx1), 
  districtTbl(t.districtTbl), districtTblIdx0(t.districtTblIdx0), 
  orderLineTbl(t.orderLineTbl), orderLineTblIdx0(t.orderLineTblIdx0), orderLineTblIdx1(t.orderLineTblIdx1), 
  customerTbl(t.customerTbl), customerTblIdx0(t.customerTblIdx0), customerTblIdx1(t.customerTblIdx1), 
  stockTbl(t.stockTbl), stockTblIdx0(t.stockTblIdx0){
     memset(threadXactCounts, 0, sizeof(uint)*5);
  }
  
  newOrderTblStoreType& newOrderTbl;  newOrderTblIdx0Type& newOrderTblIdx0;  newOrderTblIdx1Type& newOrderTblIdx1;
  historyTblStoreType& historyTbl;  historyTblIdx0Type& historyTblIdx0;
  warehouseTblStoreType& warehouseTbl;  warehouseTblIdx0Type& warehouseTblIdx0;
  itemTblStoreType& itemTbl;  itemTblIdx0Type& itemTblIdx0;
  orderTblStoreType& orderTbl;  orderTblIdx0Type& orderTblIdx0;  orderTblIdx1Type& orderTblIdx1;
  districtTblStoreType& districtTbl;  districtTblIdx0Type& districtTblIdx0;
  orderLineTblStoreType& orderLineTbl;  orderLineTblIdx0Type& orderLineTblIdx0;  orderLineTblIdx1Type& orderLineTblIdx1;
  customerTblStoreType& customerTbl;  customerTblIdx0Type& customerTblIdx0;  customerTblIdx1Type& customerTblIdx1;
  stockTblStoreType& stockTbl;  stockTblIdx0Type& stockTblIdx0;
  
  
  
  FORCE_INLINE TransactionReturnStatus PaymentTx(Transaction& xact, int x16, date x17, int x18, int x19, int x20, int x21, int x22, int x23, int x24, PString x25, double x26) {
    struct SEntry9_ISSSSSSDD y14639; struct SEntry9_ISSSSSSDD* x14639 = &y14639;
    x14639->_1 = x19;
    OperationReturnStatus stx7667;
    struct SEntry9_ISSSSSSDD* x7667 =  warehouseTblIdx0.getForUpdate(x14639, stx7667, xact);
    if(stx7667 == WW_VALUE) return WW_ABORT;
    double x3299 = x7667->_9;
    x7667->_9 = (x3299+(x26));
    warehouseTblIdx0.update(x7667);
    struct SEntry11_IISSSSSSDDI y14647; struct SEntry11_IISSSSSSDDI* x14647 = &y14647;
    x14647->_1 = x20; x14647->_2 = x19;
    OperationReturnStatus stx7674;
    struct SEntry11_IISSSSSSDDI* x7674 =  districtTblIdx0.getForUpdate(x14647, stx7674, xact);
    if(stx7674 == WW_VALUE) return WW_ABORT;
    double x3306 = x7674->_10;
    x7674->_10 = (x3306+(x26));
    districtTblIdx0.update(x7674);
    struct SEntry21_IIISSSSSSSSSTSDDDDIIS* ite13976 = 0;
    if((x21>(0))) {
      std::vector<struct SEntry21_IIISSSSSSSSSTSDDDDIIS*> x13980results;
      MedianAggregator<struct SEntry21_IIISSSSSSSSSTSDDDDIIS, PString> x13980([&](struct SEntry21_IIISSSSSSSSSTSDDDDIIS* e) -> PString {
        return (e->_4); 
      }, x13980results);
      struct SEntry21_IIISSSSSSSSSTSDDDDIIS y14662; struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x14662 = &y14662;
      x14662->_2 = x23; x14662->_3 = x22; x14662->_6 = x25;
      customerTblIdx1.sliceNoUpdate(x14662, x13980, xact);
      OperationReturnStatus stx13983;
      struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x13983 = x13980.resultForUpdate(stx13983, xact);
      if(stx13983 == WW_VALUE) return WW_ABORT;
      ite13976 = x13983;
    } else {
      struct SEntry21_IIISSSSSSSSSTSDDDDIIS y14668; struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x14668 = &y14668;
      x14668->_1 = x24; x14668->_2 = x23; x14668->_3 = x22;
      OperationReturnStatus stx13986;
      struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x13986 =  customerTblIdx0.getForUpdate(x14668, stx13986, xact);
      if(stx13986 == WW_VALUE) return WW_ABORT;
      ite13976 = x13986;
    };
    struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x3311 = ite13976;
    char* x15068 = strstr((x3311->_14).data_, "BC");
    if((x15068!=(NULL))) {
      PString c_new_data(500);
      snprintf(c_new_data.data_, 501, "%d %d %d %d %d $%f %s | %s", (x3311->_1), x23, x22, x20, x19, x26, IntToStrdate(x17), (x3311->_21).data_);
      double x3319 = x3311->_17;
      x3311->_17 = (x3319+(x26));
      x3311->_21 = c_new_data;
    } else {
      double x3323 = x3311->_17;
      x3311->_17 = (x3323+(x26));
    };
    customerTblIdx1.update(x3311);
    customerTblIdx0.update(x3311);
    PString h_data(24);
    snprintf(h_data.data_, 25, "%.10s    %.10s", (x7667->_2).data_, (x7674->_3).data_);
    struct SEntry8_IIIIITDS y14695; struct SEntry8_IIIIITDS* x14695 = &y14695;
    x14695->_1 = (x3311->_1); x14695->_2 = x23; x14695->_3 = x22; x14695->_4 = x20; x14695->_5 = x19; x14695->_6 = x17; x14695->_7 = x26; x14695->_8 = h_data;
    OperationReturnStatus st13036 = historyTbl.insert_nocheck(x14695, xact);
    if(st13036 == WW_VALUE) return WW_ABORT;
    clearTempMem();
    return SUCCESS;
  }
  
  FORCE_INLINE TransactionReturnStatus NewOrderTx(Transaction& xact, int x78, date x79, int x80, int x81, int x82, int x83, int x84, int x85, int* x86, int* x87, int* x88, double* x89, PString* x90, int* x91, PString* x92, double* x93) {
    int x95 = 0;
    int x98 = 0;
    PString idata[x84];
    int x103 = 1;
    int x106 = 1;
    while(1) {
      
      int x108 = x95;
      int ite14188 = 0;
      if((x108<(x84))) {
        
        int x110 = x103;
        int x14189 = x110;
        ite14188 = x14189;
      } else {
        ite14188 = 0;
      };
      int x14071 = ite14188;
      if (!(x14071)) break; 
      
      int x113 = x95;
      int supwid = x87[x113];
      if((supwid!=(x81))) {
        x106 = 0;
      };
      int x119 = x95;
      int x120 = x86[x119];
      struct SEntry5_IISDS y14721; struct SEntry5_IISDS* x14721 = &y14721;
      x14721->_1 = x120;
      struct SEntry5_IISDS* x6411 = itemTblIdx0.get(x14721, xact);
      if((x6411==(NULL))) {
        x103 = 0;
      } else {
        int x126 = x95;
        x90[x126] = (x6411->_3);
        int x129 = x95;
        x89[x129] = (x6411->_4);
        int x132 = x95;
        idata[x132] = (x6411->_5);
      };
      int x136 = x95;
      x95 = (x136+(1));
    };
    int x140 = x103;
    if(x140) {
      struct SEntry21_IIISSSSSSSSSTSDDDDIIS y14742; struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x14742 = &y14742;
      x14742->_1 = x83; x14742->_2 = x82; x14742->_3 = x81;
      struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x6431 = customerTblIdx0.get(x14742, xact);
      struct SEntry9_ISSSSSSDD y14746; struct SEntry9_ISSSSSSDD* x14746 = &y14746;
      x14746->_1 = x81;
      struct SEntry9_ISSSSSSDD* x6434 = warehouseTblIdx0.get(x14746, xact);
      struct SEntry11_IISSSSSSDDI y14750; struct SEntry11_IISSSSSSDDI* x14750 = &y14750;
      x14750->_1 = x82; x14750->_2 = x81;
      OperationReturnStatus stx7863;
      struct SEntry11_IISSSSSSDDI* x7863 =  districtTblIdx0.getForUpdate(x14750, stx7863, xact);
      if(stx7863 == WW_VALUE) return WW_ABORT;
      int x3529 = x7863->_11;
      int x3530 = x7863->_11;
      x7863->_11 = (x3530+(1));
      districtTblIdx0.update(x7863);
      int x157 = x106;
      struct SEntry8_IIIITIIB y14760; struct SEntry8_IIIITIIB* x14760 = &y14760;
      x14760->_1 = x3529; x14760->_2 = x82; x14760->_3 = x81; x14760->_4 = x83; x14760->_5 = x79; x14760->_6 = -1; x14760->_7 = x84; x14760->_8 = x157;
      OperationReturnStatus st13086 = orderTbl.insert_nocheck(x14760, xact);
      if(st13086 == WW_VALUE) return WW_ABORT;
      struct SEntry3_III y14764; struct SEntry3_III* x14764 = &y14764;
      x14764->_1 = x3529; x14764->_2 = x82; x14764->_3 = x81;
      OperationReturnStatus st13088 = newOrderTbl.insert_nocheck(x14764, xact);
      if(st13088 == WW_VALUE) return WW_ABORT;
      double x165 = 0.0;
      x95 = 0;
      while(1) {
        
        int x168 = x95;
        if (!((x168<(x84)))) break; 
        
        int x171 = x95;
        int ol_supply_w_id = x87[x171];
        int x174 = x95;
        int ol_i_id = x86[x174];
        int x177 = x95;
        int ol_quantity = x88[x177];
        struct SEntry17_IIISSSSSSSSSSIIIS y14779; struct SEntry17_IIISSSSSSSSSSIIIS* x14779 = &y14779;
        x14779->_1 = ol_i_id; x14779->_2 = ol_supply_w_id;
        OperationReturnStatus stx7889;
        struct SEntry17_IIISSSSSSSSSSIIIS* x7889 =  stockTblIdx0.getForUpdate(x14779, stx7889, xact);
        if(stx7889 == WW_VALUE) return WW_ABORT;
        const PString& x3556 = *(&x7889->_4 + (x82-1));
        int x3557 = x7889->_3;
        int x188 = x95;
        x91[x188] = x3557;
        int x190 = x95;
        PString& x191 = idata[x190];
        char* x15243 = strstr(x191.data_, "original");
        int ite14257 = 0;
        if((x15243!=(NULL))) {
          
          char* x15249 = strstr((x7889->_17).data_, "original");
          int x14258 = (x15249!=(NULL));
          ite14257 = x14258;
        } else {
          ite14257 = 0;
        };
        int x14135 = ite14257;
        if(x14135) {
          int x196 = x95;
          x92[x196].data_[0] = 'B';
        } else {
          int x198 = x95;
          x92[x198].data_[0] = 'G';
        };
        x7889->_3 = (x3557-(ol_quantity));
        if((x3557<=(ol_quantity))) {
          int x3575 = x7889->_3;
          x7889->_3 = (x3575+(91));
        };
        int x207 = 0;
        if((ol_supply_w_id!=(x81))) {
          x207 = 1;
        };
        stockTblIdx0.update(x7889);
        int x220 = x95;
        double x221 = x89[x220];
        double ol_amount = ((ol_quantity*(x221))*(((1.0+((x6434->_8)))+((x7863->_9)))))*((1.0-((x6431->_16))));
        int x229 = x95;
        x93[x229] = ol_amount;
        double x231 = x165;
        x165 = (x231+(ol_amount));
        int x234 = x95;
        struct SEntry10_IIIIIITIDS y14833; struct SEntry10_IIIIIITIDS* x14833 = &y14833;
        x14833->_1 = x3529; x14833->_2 = x82; x14833->_3 = x81; x14833->_4 = (x234+(1)); x14833->_5 = ol_i_id; x14833->_6 = ol_supply_w_id; x14833->_8 = ol_quantity; x14833->_9 = ol_amount; x14833->_10 = x3556;
        OperationReturnStatus st13150 = orderLineTbl.insert_nocheck(x14833, xact);
        if(st13150 == WW_VALUE) return WW_ABORT;
        int x239 = x95;
        x95 = (x239+(1));
      };
    } else {
      int x243 = failedNO;
      failedNO = (1+(x243));
    };
    clearTempMem();
    return SUCCESS;
  }
  
  FORCE_INLINE TransactionReturnStatus DeliveryTx(Transaction& xact, int x247, date x248, int x249, int x250) {
    int orderIDs[10];
    int x255 = 1;
    while(1) {
      
      int x257 = x255;
      if (!((x257<=(10)))) break; 
      
      struct SEntry3_III* x3745result = nullptr;
      MinAggregator<struct SEntry3_III, int> x3745([&](struct SEntry3_III* e) -> int {
        return (e->_1); 
      }, &x3745result);
      int x264 = x255;
      struct SEntry3_III y14853; struct SEntry3_III* x14853 = &y14853;
      x14853->_2 = x264; x14853->_3 = x249;
      newOrderTblIdx1.sliceNoUpdate(x14853, x3745, xact);
      OperationReturnStatus stx8087;
      struct SEntry3_III* x8087 = x3745.resultForUpdate(stx8087, xact);
      if(stx8087 == WW_VALUE) return WW_ABORT;
      if((x8087!=(NULL))) {
        int x3753 = x8087->_1;
        int x273 = x255;
        orderIDs[(x273-(1))] = x3753;
        newOrderTbl.del(x8087);
        int x278 = x255;
        struct SEntry8_IIIITIIB y14866; struct SEntry8_IIIITIIB* x14866 = &y14866;
        x14866->_1 = x3753; x14866->_2 = x278; x14866->_3 = x249;
        OperationReturnStatus stx8098;
        struct SEntry8_IIIITIIB* x8098 =  orderTblIdx0.getForUpdate(x14866, stx8098, xact);
        if(stx8098 == WW_VALUE) return WW_ABORT;
        x8098->_6 = x250;
        orderTblIdx0.update(x8098);
        orderTblIdx1.update(x8098);
        double x287 = 0.0;
        int x289 = x255;
        struct SEntry10_IIIIIITIDS y14885; struct SEntry10_IIIIIITIDS* x14885 = &y14885;
        x14885->_1 = x3753; x14885->_2 = x289; x14885->_3 = x249;
        OperationReturnStatus st6690 = orderLineTblIdx1.slice(x14885, [&](struct SEntry10_IIIIIITIDS* orderLineEntry) -> TransactionReturnStatus {
          orderLineEntry->_7 = x248;
          double x294 = x287;
          x287 = (x294+((orderLineEntry->_9)));
          orderLineTblIdx1.update(orderLineEntry);
          orderLineTblIdx0.update(orderLineEntry);
          return SUCCESS;
        }, xact);
        if(st6690 == WW_VALUE) return WW_ABORT;
        int x302 = x255;
        struct SEntry21_IIISSSSSSSSSTSDDDDIIS y14890; struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x14890 = &y14890;
        x14890->_1 = (x8098->_4); x14890->_2 = x302; x14890->_3 = x249;
        OperationReturnStatus stx8120;
        struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x8120 =  customerTblIdx0.getForUpdate(x14890, stx8120, xact);
        if(stx8120 == WW_VALUE) return WW_ABORT;
        double x306 = x287;
        double x3776 = x8120->_17;
        x8120->_17 = (x3776+(x306));
        int x3779 = x8120->_20;
        x8120->_20 = (x3779+(1));
        customerTblIdx1.update(x8120);
        customerTblIdx0.update(x8120);
      } else {
        int x310 = failedDel;
        failedDel = (1+(x310));
        int x313 = x255;
        orderIDs[(x313-(1))] = 0;
      };
      int x317 = x255;
      x255 = (x317+(1));
    };
    clearTempMem();
    return SUCCESS;
  }
  
  FORCE_INLINE TransactionReturnStatus StockLevelTx(Transaction& xact, int x321, date x322, int x323, int x324, int x325, int x326) {
    struct SEntry11_IISSSSSSDDI y14912; struct SEntry11_IISSSSSSDDI* x14912 = &y14912;
    x14912->_1 = x325; x14912->_2 = x324;
    struct SEntry11_IISSSSSSDDI* x6780 = districtTblIdx0.get(x14912, xact);
    int x3919 = x6780->_11;
    int x335 = (x3919-(20));
    unordered_set<int> unique_ol_i_id; //setApply2
    while(1) {
      
      int x339 = x335;
      if (!((x339<(x3919)))) break; 
      
      int x341 = x335;
      struct SEntry10_IIIIIITIDS y14935; struct SEntry10_IIIIIITIDS* x14935 = &y14935;
      x14935->_1 = x341; x14935->_2 = x325; x14935->_3 = x324;
      OperationReturnStatus st9412 = orderLineTblIdx1.sliceNoUpdate(x14935, [&](struct SEntry10_IIIIIITIDS* orderLineEntry) -> TransactionReturnStatus {
        int x3947 = orderLineEntry->_5;
        struct SEntry17_IIISSSSSSSSSSIIIS y14926; struct SEntry17_IIISSSSSSSSSSIIIS* x14926 = &y14926;
        x14926->_1 = x3947; x14926->_2 = x324;
        struct SEntry17_IIISSSSSSSSSSIIIS* x6793 = stockTblIdx0.get(x14926, xact);
        if(((x6793->_3)<(x326))) {
          unique_ol_i_id.insert(x3947);
        };
        return SUCCESS;
      }, xact);
      if(st9412 == WW_VALUE) return WW_ABORT;
      int x358 = x335;
      x335 = (x358+(1));
    };
    clearTempMem();
    return SUCCESS;
  }
  
  FORCE_INLINE TransactionReturnStatus OrderStatusTx(Transaction& xact, int x364, date x365, int x366, int x367, int x368, int x369, int x370, PString x371) {
    struct SEntry21_IIISSSSSSSSSTSDDDDIIS* ite14522 = 0;
    if((x369>(0))) {
      std::vector<struct SEntry21_IIISSSSSSSSSTSDDDDIIS*> x14526results;
      MedianAggregator<struct SEntry21_IIISSSSSSSSSTSDDDDIIS, PString> x14526([&](struct SEntry21_IIISSSSSSSSSTSDDDDIIS* e) -> PString {
        return (e->_4); 
      }, x14526results);
      struct SEntry21_IIISSSSSSSSSTSDDDDIIS y14949; struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x14949 = &y14949;
      x14949->_2 = x368; x14949->_3 = x367; x14949->_6 = x371;
      customerTblIdx1.sliceNoUpdate(x14949, x14526, xact);
      struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x14529 = x14526.result();
      ite14522 = x14529;
    } else {
      struct SEntry21_IIISSSSSSSSSTSDDDDIIS y14955; struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x14955 = &y14955;
      x14955->_1 = x370; x14955->_2 = x368; x14955->_3 = x367;
      struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x14532 = customerTblIdx0.get(x14955, xact);
      ite14522 = x14532;
    };
    struct SEntry21_IIISSSSSSSSSTSDDDDIIS* x3992 = ite14522;
    struct SEntry8_IIIITIIB* x3995result = nullptr;
    MaxAggregator<struct SEntry8_IIIITIIB, int> x3995([&](struct SEntry8_IIIITIIB* e) -> int {
      return (e->_1); 
    }, &x3995result);
    struct SEntry8_IIIITIIB y14966; struct SEntry8_IIIITIIB* x14966 = &y14966;
    x14966->_2 = x368; x14966->_3 = x367; x14966->_4 = (x3992->_3);
    orderTblIdx1.sliceNoUpdate(x14966, x3995, xact);
    struct SEntry8_IIIITIIB* x3999 = x3995.result();
    int ite14545 = 0;
    if((x3999==(NULL))) {
      int x14546 = failedOS;
      failedOS = (1+(x14546));
      ite14545 = 0;
    } else {
      struct SEntry10_IIIIIITIDS y14982; struct SEntry10_IIIIIITIDS* x14982 = &y14982;
      x14982->_1 = (x3999->_1); x14982->_2 = x368; x14982->_3 = x367;
      OperationReturnStatus st14555 = orderLineTblIdx1.sliceNoUpdate(x14982, [&](struct SEntry10_IIIIIITIDS* orderLineEntry) -> TransactionReturnStatus {
        int x409 = 1;
        return SUCCESS;
      }, xact);
      if(st14555 == WW_VALUE) return WW_ABORT;
      ite14545 = 1;
    };
    int x413 = ite14545;
    clearTempMem();
    return SUCCESS;
  }
  
   TransactionReturnStatus runProgram(Program* prg);
};

TransactionManager xactManager;
TransactionManager& Transaction::tm(xactManager);
uint globalXactCounts[5];
uint8_t prgId7to5[] = {0, 1, 1, 2, 2, 3, 4};

volatile bool isReady[numThreads];
volatile bool startExecution, hasFinished;


TPCC_Data orig;
#ifdef VERIFY_CONC
   TPCC_Data res;
#endif
 

#include "sc/TPCC.h"

TPCCDataGen tpcc;

TransactionReturnStatus ThreadLocal::runProgram(Program* prg) {
  TransactionReturnStatus ret = SUCCESS;
  switch (prg->id) {
    case NEWORDER:
    {
      NewOrder& p = *(NewOrder *) prg;
      ret = NewOrderTx(prg->xact, false, p.datetime, -1, p.w_id, p.d_id, p.c_id, p.o_ol_cnt, p.o_all_local, p.itemid, p.supware, p.quantity, p.price, p.iname, p.stock, p.bg, p.amt);
      break;
    }
    case PAYMENTBYID:
    {
      PaymentById& p = *(PaymentById *) prg;
      ret = PaymentTx(prg->xact, false, p.datetime, -1, p.w_id, p.d_id, 0, p.c_w_id, p.c_d_id, p.c_id, nullptr, p.h_amount);
      break;
    }
    case PAYMENTBYNAME:
    {
      PaymentByName& p = *(PaymentByName *) prg;
      ret = PaymentTx(prg->xact, false, p.datetime, -1, p.w_id, p.d_id, 1, p.c_w_id, p.c_d_id, -1, p.c_last_input, p.h_amount);
      break;
    }
    case ORDERSTATUSBYID:
    {
      OrderStatusById &p = *(OrderStatusById *) prg;
      ret = OrderStatusTx(prg->xact, false, -1, -1, p.w_id, p.d_id, 0, p.c_id, nullptr);
      break;
    }
    case ORDERSTATUSBYNAME:
    {
      OrderStatusByName &p = *(OrderStatusByName *) prg;
      ret = OrderStatusTx(prg->xact, false, -1, -1, p.w_id, p.d_id, 1, -1, p.c_last);
      break;
    }
    case DELIVERY:
    {
      Delivery &p = *(Delivery *) prg;
      ret = DeliveryTx(prg->xact, false, p.datetime, p.w_id, p.o_carrier_id);
      break;
    }
    case STOCKLEVEL:
    {
      StockLevel &p = *(StockLevel *) prg;
      ret = StockLevelTx(prg->xact, false, -1, -1, p.w_id, p.d_id, p.threshold);
      break;
    }
    default: cerr << "UNKNOWN PROGRAM TYPE" << endl;

  }
  return ret;
}
      
std::atomic<uint> PC(0);
void threadFunction(uint8_t thread_id, ThreadLocal* tl) {
    setAffinity(thread_id);
    //    setSched(SCHED_FIFO);


  isReady[thread_id] = true;
  uint pid = PC++;
  Program* p;
  TransactionReturnStatus st;
  while (!startExecution);
  const uint failedProgramSize = 32;
  Program * failedPrograms[failedProgramSize];
  uint head = 0, tail = 0;
  bool full = false;
  p = tpcc.programs[pid];
  while (!hasFinished) {

    xactManager.begin(p->xact, thread_id);

    st = tl->runProgram(p);

    if (st != SUCCESS) {
      xactManager.rollback(p->xact, thread_id);
      if (!full && p->xact.failedBecauseOf != nullptr) {
        failedPrograms[tail++] = p;
        if (tail == failedProgramSize)
          tail = 0;
        if (head == tail)
          full = true;
        pid = PC++;
        if (pid >= numPrograms)
          break;
        p = tpcc.programs[pid];
      }
    } else {
      if (xactManager.validateAndCommit(p->xact, thread_id)) {   //rollback happens inside function if it fails
        tl->threadXactCounts[prgId7to5[p->id]]++;
        if (head != tail || full) {
          p = failedPrograms[head];
          if (p->xact.failedBecauseOf->commitTS != initCommitTS) {
            head++;
            full = false;
            if (head == failedProgramSize)
              head = 0;
            continue;
          }
        }
        pid = PC++;
        if(pid >= numPrograms)
          break;
        p = tpcc.programs[pid];
      }
    }
  }
#ifndef VERIFY_CONC
  hasFinished = true;
#endif
}

       
/* TRAITS STARTING */


int main(int argc, char** argv) {
 /* TRAITS ENDING   */
  
  setAffinity(-1);
  #ifndef NORESIZE
  cout << "Index Resizing warning disabled" << endl;
  #endif
  
  cout  << "NumThreads = " << numThreads << endl;
  
  tpcc.loadPrograms();
  
  Transaction t0;
  xactManager.begin(t0, 0);
  tpcc.loadWare(t0);
  tpcc.loadDist(t0);
  tpcc.loadCust(t0);
  tpcc.loadItem(t0);
  tpcc.loadNewOrd(t0);
  tpcc.loadOrders(t0);
  tpcc.loadOrdLine(t0);
  tpcc.loadHist(t0);
  tpcc.loadStocks(t0);
  xactManager.commit(t0, 0);
  cout.imbue(std::locale(""));
  
  memset(globalXactCounts, 0, 5 * sizeof(uint));
  memset(xactManager.activeXactStartTS, 0xff, sizeof(xactManager.activeXactStartTS[0]) * numThreads);
  ThreadLocal *tls[numThreads];
  Timepoint startTime, endTime;
  std::thread workers[numThreads];
  
  for (uint8_t i = 0; i < numThreads; ++i) {
      tls[i] = new ThreadLocal(i, orig);
      workers[i] = std::thread(threadFunction, i, tls[i]);
  }
  bool all_ready = true;
  //check if all worker threads are ready. Execution can be started once all threads finish startup procedure
  while (true) {
      for (uint8_t i = 0; i < numThreads; ++i) {
          if (isReady[i] == false) {
              all_ready = false;
              break;
          }
      }
      if (all_ready) {
          startTime = Now;
          startExecution = true;
          break;
      }
      all_ready = true;
  }
  
  for (uint8_t i = 0; i < numThreads; ++i) {
      workers[i].join();
  }
  endTime = Now;
  auto execTime = DurationMS(endTime - startTime);
  
  for(uint i = 0; i < numThreads;  ++i) {
     for(uint j = 0; j < 5; ++j) {
        globalXactCounts[j] += tls[i]->threadXactCounts[j];
      }
  }
  
  cout << "Failed NO = " << failedNO << endl;
  cout << "Failed Del = " << failedDel << endl;
  cout << "Failed OS = " << failedOS << endl;
  cout << "Total time = " << execTime << " ms" << endl;
  cout << "Total transactions = " << numPrograms << "   NewOrder = " <<  globalXactCounts[0]  << endl;
  cout << "TpmC = " << fixed <<  (globalXactCounts[0])* 60000.0/execTime << endl;
  
  ofstream fout("tpcc_res_cpp.csv", ios::app);
  if(argc == 1 || atoi(argv[1]) == 1) {
    fout << "\nCPP-EFINR-" << numPrograms << ",";
    for(int i = 0; i < 5 ; ++i)
       fout << globalXactCounts[i] << ",";
    fout <<",";
   }
  fout << execTime << ",";
  fout.close();
  
  /*
  ofstream info("/Users/milllic/Documents/repo/dbtoaster-backend/runtime/stats/default.json");
  info << "{\n";
  GET_RUN_STAT(orig.newOrderTblIdx0, info);
  info <<",\n";
  GET_RUN_STAT(orig.newOrderTblIdx1, info);
  info <<",\n";
  GET_RUN_STAT(orig.orderLineTblIdx0, info);
  info <<",\n";
  GET_RUN_STAT(orig.orderLineTblIdx1, info);
  info <<",\n";
  GET_RUN_STAT(orig.itemTblIdx0, info);
  info <<",\n";
  GET_RUN_STAT(orig.historyTblIdx0, info);
  info <<",\n";
  GET_RUN_STAT(orig.districtTblIdx0, info);
  info <<",\n";
  GET_RUN_STAT(orig.customerTblIdx0, info);
  info <<",\n";
  GET_RUN_STAT(orig.customerTblIdx1, info);
  info <<",\n";
  GET_RUN_STAT(orig.stockTblIdx0, info);
  info <<",\n";
  GET_RUN_STAT(orig.orderTblIdx0, info);
  info <<",\n";
  GET_RUN_STAT(orig.orderTblIdx1, info);
  info <<",\n";
  GET_RUN_STAT(orig.warehouseTblIdx0, info);
  info << "\n}\n";
  info.close();
  */
  
  #ifdef VERIFY_CONC
  ThreadLocal ver(-1, res);
  std::sort(tpcc.programs, tpcc.programs + numPrograms, [](const Program* a, const Program * b) {
    return a->xact.commitTS < b->xact.commitTS;
  });
  xactManager.committedXactsTail = nullptr;
  for (uint i = 0; i < numPrograms; ++i) {
    Program* p = tpcc.programs[i];
    if (p->xact.commitTS == initCommitTS)
      break;
    p->xact.reset();
    TransactionReturnStatus st;
    xactManager.begin(p->xact, 0);
    st = ver.runProgram(p);
    assert(st == SUCCESS);
    bool st2 = xactManager.validateAndCommit(p->xact, 0);
    assert(st2);
  }
  
  if (orig.warehouseTblIdx0 == res.warehouseTblIdx0) {
    cout << "Warehouse results are same as serial version" << endl;
  } else {
    cerr << "Warehouse results INCORRECT!" << endl;
  }
  if (orig.districtTblIdx0 == res.districtTblIdx0) {
    cout << "District results are same as serial version" << endl;
  } else {
    cerr << "District results INCORRECT!" << endl;
  }
  if (orig.customerTblIdx0 == res.customerTblIdx0) {
    cout << "Customer results are same as serial version" << endl;
  } else {
    cerr << "Customer results INCORRECT!" << endl;
  }
  if (orig.orderTblIdx0 == res.orderTblIdx0) {
    cout << "Order results are same as serial version" << endl;
  } else {
    cerr << "Order results INCORRECT!" << endl;
  }
  if (orig.orderLineTblIdx0 == res.orderLineTblIdx0) {
    cout << "OrderLine results are same as serial version" << endl;
  } else {
    cerr << "OrderLine results INCORRECT!" << endl;
  }
  if (orig.newOrderTblIdx0 == res.newOrderTblIdx0) {
    cout << "NewOrder results are same as serial version" << endl;
  } else {
    cerr << "NewOrder results INCORRECT!" << endl;
  }
  if (orig.itemTblIdx0 == res.itemTblIdx0) {
    cout << "Item results are same as serial version" << endl;
  } else {
    cerr << "Item results INCORRECT!" << endl;
  }
  if (orig.stockTblIdx0 == res.stockTblIdx0) {
    cout << "Stock results are same as serial version" << endl;
  } else {
    cerr << "Stock results INCORRECT!" << endl;
  }
  if (orig.historyTblIdx0 == res.historyTblIdx0) {
    cout << "History results are same as serial version" << endl;
  } else {
    cerr << "History results INCORRECT!" << endl;
  }
  #endif
  #ifdef VERIFY_TPCC
  /*
      warehouseTblIdx0.resize_(warehouseTblSize); tpcc.wareRes.resize_(warehouseTblSize);
      districtTblIdx0.resize_(districtTblSize); tpcc.distRes.resize_(districtTblSize);
      customerTblIdx0.resize_(customerTblSize); tpcc.custRes.resize_(customerTblSize);
      orderTblIdx0.resize_(orderTblSize); tpcc.ordRes.resize_(orderTblSize);
      newOrderTblIdx0.resize_(newOrderTblSize); tpcc.newOrdRes.resize_(newOrderTblSize);
      orderLineTblIdx0.resize_(orderLineTblSize); tpcc.ordLRes.resize_(orderLineTblSize);
      itemTblIdx0.resize_(itemTblSize); tpcc.itemRes.resize_(itemTblSize);
      stockTblIdx0.resize_(stockTblSize); tpcc.stockRes.resize_(stockTblSize);
      historyTblIdx0.resize_(historyTblSize); tpcc.histRes.resize_(historyTblSize);
  */
      if (orig.warehouseTblIdx0 == tpcc.wareRes) {
          cout << "Warehouse results are correct" << endl;
      } else {
          cerr << "Warehouse results INCORRECT!" << endl;
      }
      if (orig.districtTblIdx0 == tpcc.distRes) {
          cout << "District results are correct" << endl;
      } else {
          cerr << "District results INCORRECT!" << endl;
      }
      if (orig.customerTblIdx0 == tpcc.custRes) {
          cout << "Customer results are correct" << endl;
      } else {
          cerr << "Customer results INCORRECT!" << endl;
      }
      if (orig.orderTblIdx0 == tpcc.ordRes) {
          cout << "Order results are correct" << endl;
      } else {
          cerr << "Order results INCORRECT!" << endl;
      }
      if (orig.orderLineTblIdx0 == tpcc.ordLRes) {
          cout << "OrderLine results are correct" << endl;
      } else {
          cerr << "OrderLine results INCORRECT!" << endl;
      }
      if (orig.newOrderTblIdx0 == tpcc.newOrdRes) {
          cout << "NewOrder results are correct" << endl;
      } else {
          cerr << "NewOrder results INCORRECT!" << endl;
      }
      if (orig.itemTblIdx0 == tpcc.itemRes) {
          cout << "Item results are correct" << endl;
      } else {
          cerr << "Item results INCORRECT!" << endl;
      }
      if (orig.stockTblIdx0 == tpcc.stockRes) {
          cout << "Stock results are correct" << endl;
      } else {
          cerr << "Stock results INCORRECT!" << endl;
      }
      if (orig.historyTblIdx0 == tpcc.histRes) {
          cout << "History results are correct" << endl;
      } else {
          cerr << "History results INCORRECT!" << endl;
      }
  
  #endif
  
        
}
