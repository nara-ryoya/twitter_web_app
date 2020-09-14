import sympy 

class SentimentAnalysis(object):
    #テストで今日の日付を出力するプログラムを生成（ここで分析を行う）
    @classmethod
    def prime_factorize(cls, n):
        a = []
        while n % 2 == 0:
            a.append(2)
            n //= 2
        f = 3
        while f * f <= n:
            if n % f == 0:
                a.append(f)
                n //= f
            else:
                f += 2
        if n != 1:
            a.append(n)
        return a

    @classmethod
    def isprime(cls, num):
        a = cls.prime_factorize(num)

        if sympy.isprime(num):
            return f"素数です:{a}"
        else:
            return f"素数ではないです:{a}"


