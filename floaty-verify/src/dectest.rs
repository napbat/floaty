//! Reads decTest files, the decimal arithmetic test vectors of Mike
//! Cowlishaw.
//!
//! `build.rs` extracts the pinned decTest archive, version 2.62, into
//! [`DIRECTORY`]. The `dd` files test decimal64, `decDouble` in decNumber;
//! the `dq` files test decimal128, `decQuad`; and the `ds` files test
//! decimal32, `decSingle`. The document "General Decimal Arithmetic
//! Testcases" on speleotrove.com defines the format.
//!
//! [`parse`] reads the text of a file into [`Test`]s. [`Test::encode`]
//! converts the operands and the result of a test to the encodings of one
//! format through decNumber, because floaty does not convert strings.

use std::fmt;
use std::fs;
use std::path::PathBuf;

use crate::decnumber::{Binary, Format, Rounding, Status, Unary};

/// The directory that holds the extracted `.decTest` files.
pub const DIRECTORY: &str = env!("FLOATY_DECTEST_DIR");

/// The settings of the directives that apply to a test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Context {
    /// The working precision in digits.
    pub precision: u32,
    /// The rounding mode.
    pub rounding: Rounding,
    /// The largest adjusted exponent.
    pub max_exponent: i32,
    /// The smallest adjusted exponent of a normal value.
    pub min_exponent: i32,
    /// Whether the tests need the full arithmetic, not the X3.274 subset.
    pub extended: bool,
    /// Whether the exponent clamps as in the IEEE 754 interchange formats.
    pub clamp: bool,
}

impl Context {
    /// Returns whether the context is the context of a fixed-size format,
    /// with the full arithmetic.
    #[must_use]
    pub fn fits<F: Format>(&self) -> bool {
        self.precision == F::PRECISION
            && self.max_exponent == F::EMAX
            && self.min_exponent == F::EMIN
            && self.clamp
            && self.extended
    }
}

/// The operation of a test.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    /// An operation with one operand.
    Unary(Unary),
    /// An operation with two operands.
    Binary(Binary),
    /// `fma`: a fused multiply-add of three operands.
    Fma,
    /// `class`: the class name of the operand.
    Class,
    /// `samequantum`: `1` when the operands have the same exponent, else `0`.
    SameQuantum,
    /// `apply`: the conversion of the operand to a number in the context.
    Apply,
    /// `tosci`: the conversion to a number and back to a scientific string.
    ToSci,
    /// `toeng`: the conversion to a number and back to an engineering string.
    ToEng,
    /// An operation that the `dd`, `dq`, and `ds` files do not use, in lower
    /// case.
    Other(String),
}

impl Operation {
    /// Returns the operation with a decTest name. The name is
    /// case-independent.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        let unary = match lower.as_str() {
            "abs" => Some(Unary::Abs),
            "canonical" => Some(Unary::Canonical),
            "copy" => Some(Unary::Copy),
            "copyabs" => Some(Unary::CopyAbs),
            "copynegate" => Some(Unary::CopyNegate),
            "invert" => Some(Unary::Invert),
            "logb" => Some(Unary::LogB),
            "minus" => Some(Unary::Minus),
            "nextminus" => Some(Unary::NextMinus),
            "nextplus" => Some(Unary::NextPlus),
            "plus" => Some(Unary::Plus),
            "reduce" => Some(Unary::Reduce),
            "tointegralx" => Some(Unary::ToIntegralExact),
            _ => None,
        };
        if let Some(unary) = unary {
            return Self::Unary(unary);
        }
        if let Some(binary) = binary_from_name(&lower) {
            return Self::Binary(binary);
        }
        match lower.as_str() {
            "fma" => Self::Fma,
            "class" => Self::Class,
            "samequantum" => Self::SameQuantum,
            "apply" => Self::Apply,
            "tosci" => Self::ToSci,
            "toeng" => Self::ToEng,
            _ => Self::Other(lower),
        }
    }

    /// Returns the operand count of a known operation.
    #[must_use]
    pub fn arity(&self) -> Option<usize> {
        match self {
            Self::Unary(_) | Self::Class | Self::Apply | Self::ToSci | Self::ToEng => Some(1),
            Self::Binary(_) | Self::SameQuantum => Some(2),
            Self::Fma => Some(3),
            Self::Other(_) => None,
        }
    }

    /// Returns whether the result is text, not a number: a class name, `0`
    /// or `1`, or a string that a conversion writes.
    #[must_use]
    pub fn has_text_result(&self) -> bool {
        matches!(
            self,
            Self::Class | Self::SameQuantum | Self::ToSci | Self::ToEng
        )
    }

    /// Returns whether the operation converts a string operand in the
    /// context, so the operand stays text.
    #[must_use]
    pub fn converts_text(&self) -> bool {
        matches!(self, Self::Apply | Self::ToSci | Self::ToEng)
    }
}

/// Returns the operation with two operands that has a lower-case name.
fn binary_from_name(lower: &str) -> Option<Binary> {
    Some(match lower {
        "add" => Binary::Add,
        "and" => Binary::And,
        "compare" => Binary::Compare,
        "comparesig" => Binary::CompareSignal,
        "comparetotal" => Binary::CompareTotal,
        "comparetotmag" => Binary::CompareTotalMag,
        "copysign" => Binary::CopySign,
        "divide" => Binary::Divide,
        "divideint" => Binary::DivideInteger,
        "max" => Binary::Max,
        "maxmag" => Binary::MaxMag,
        "min" => Binary::Min,
        "minmag" => Binary::MinMag,
        "multiply" => Binary::Multiply,
        "nexttoward" => Binary::NextToward,
        "or" => Binary::Or,
        "quantize" => Binary::Quantize,
        "remainder" => Binary::Remainder,
        "remaindernear" => Binary::RemainderNear,
        "rotate" => Binary::Rotate,
        "scaleb" => Binary::ScaleB,
        "shift" => Binary::Shift,
        "subtract" => Binary::Subtract,
        "xor" => Binary::Xor,
        _ => return None,
    })
}

/// One test line and the directives in force for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Test {
    /// The line number in the file, from 1.
    pub line: usize,
    /// The test identifier, for example `ddadd001`.
    pub id: String,
    /// The operation.
    pub operation: Operation,
    /// The operand tokens, with quotes removed.
    pub operands: Vec<String>,
    /// The result token, with quotes removed.
    pub result: String,
    /// The conditions that the operation must raise, and no others.
    pub conditions: Status,
    /// The directives in force.
    pub context: Context,
}

/// A test operand in the encoding of a format.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operand<B> {
    /// A number, or an explicit encoding.
    Encoding(B),
    /// A string that the operation converts: the operand of `apply`,
    /// `tosci`, or `toeng`.
    Text(String),
    /// `#`: a null reference, which only tests of a by-reference interface
    /// can use.
    Null,
}

/// The expected result of a test in the encoding of a format.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expected<B> {
    /// An explicit encoding, `#` and hexadecimal digits. The result must
    /// have these bits.
    Encoding(B),
    /// A number, in its canonical encoding. The result must be the same
    /// number: sign, coefficient, exponent, and NaN payload. A canonical
    /// result therefore has these bits.
    Number(B),
    /// The text result of an operation that [`Operation::has_text_result`].
    Text(String),
    /// `?`: an undefined result. Only the conditions count.
    Undefined,
}

/// The operands and the expected result of a test in one format.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Case<B> {
    /// The operands.
    pub operands: Vec<Operand<B>>,
    /// The expected result.
    pub expected: Expected<B>,
}

/// Why a test has no [`Case`] in a format.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncodeError {
    /// The directives differ from the parameters of the format, or select
    /// the X3.274 subset arithmetic.
    Context,
    /// An explicit encoding has the width of another format.
    Width(String),
    /// A token has a form that the `dd`, `dq`, and `ds` files do not use,
    /// such as `64#1.5`.
    Unsupported(String),
    /// The conversion of an operand raises conditions: the operand is not
    /// a number of the format.
    Operand(String, Status),
    /// The conversion of the result raises conditions: the result is not a
    /// number of the format.
    Result(String, Status),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Context => write!(formatter, "the directives do not match the format"),
            Self::Width(token) => write!(formatter, "`{token}` is an encoding of another width"),
            Self::Unsupported(token) => write!(formatter, "`{token}` has an unsupported form"),
            Self::Operand(token, status) => {
                write!(formatter, "the operand `{token}` converts with {status}")
            }
            Self::Result(token, status) => {
                write!(formatter, "the result `{token}` converts with {status}")
            }
        }
    }
}

impl std::error::Error for EncodeError {}

impl Test {
    /// Converts the operands and the result to the encodings of format `F`.
    ///
    /// A numeric operand converts with the rounding mode of the test. The
    /// conversion must be exact, as decTest requires for every operation
    /// but `apply`, `tosci`, and `toeng`, whose operands stay text. It
    /// applies the `clamp` directive, as decTest specifies: an exponent
    /// above the largest exponent of a full coefficient folds down, and the
    /// exponent of a zero clamps. The fixed-size formats do not report
    /// `Clamped` or `Rounded`, so only an inexact conversion is an error.
    ///
    /// # Errors
    ///
    /// Returns an [`EncodeError`] when the context is not the context of
    /// `F`, or when a token is not a number or an encoding of `F`.
    pub fn encode<F: Format>(&self) -> Result<Case<F::Bits>, EncodeError> {
        if !self.context.fits::<F>() {
            return Err(EncodeError::Context);
        }
        let operands = self
            .operands
            .iter()
            .map(|token| self.encode_operand::<F>(token))
            .collect::<Result<Vec<_>, _>>()?;
        let expected = self.encode_result::<F>()?;
        Ok(Case { operands, expected })
    }

    fn encode_operand<F: Format>(&self, token: &str) -> Result<Operand<F::Bits>, EncodeError> {
        if token == "#" {
            return Ok(Operand::Null);
        }
        if token.contains('#') {
            return hexadecimal::<F>(token).map(Operand::Encoding);
        }
        if self.operation.converts_text() {
            return Ok(Operand::Text(token.to_owned()));
        }
        let outcome = F::from_string(token, self.context.rounding);
        if outcome.status.is_empty() {
            Ok(Operand::Encoding(outcome.value))
        } else {
            Err(EncodeError::Operand(token.to_owned(), outcome.status))
        }
    }

    fn encode_result<F: Format>(&self) -> Result<Expected<F::Bits>, EncodeError> {
        let token = self.result.as_str();
        if token == "?" {
            return Ok(Expected::Undefined);
        }
        if self.operation.has_text_result() {
            return Ok(Expected::Text(token.to_owned()));
        }
        if token.contains('#') {
            return hexadecimal::<F>(token).map(Expected::Encoding);
        }
        let outcome = F::from_string(token, self.context.rounding);
        if outcome.status.is_empty() {
            Ok(Expected::Number(outcome.value))
        } else {
            Err(EncodeError::Result(token.to_owned(), outcome.status))
        }
    }
}

/// Decodes an explicit encoding of format `F`: `#` and 8, 16, or 32
/// hexadecimal digits.
fn hexadecimal<F: Format>(token: &str) -> Result<F::Bits, EncodeError> {
    let unsupported = || EncodeError::Unsupported(token.to_owned());
    let digits = token.strip_prefix('#').ok_or_else(unsupported)?;
    if ![8, 16, 32].contains(&digits.len()) {
        return Err(unsupported());
    }
    let value = u128::from_str_radix(digits, 16).map_err(|_| unsupported())?;
    let width = 8 * size_of::<F::Bits>();
    if digits.len() * 4 != width {
        return Err(EncodeError::Width(token.to_owned()));
    }
    F::Bits::try_from(value).map_err(|_| EncodeError::Width(token.to_owned()))
}

/// Why a file is not a valid decTest file.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParseErrorKind {
    /// A quoted token has no closing quote.
    UnclosedQuote,
    /// A directive has an unknown keyword, such as `dectest`, which
    /// includes another file.
    UnknownDirective(String),
    /// A directive has an invalid value.
    DirectiveValue(String),
    /// A test comes before a required directive.
    MissingDirective(&'static str),
    /// A test has no `->` or no result.
    Malformed,
    /// A test has the wrong operand count for its operation.
    Arity,
    /// A condition name is unknown.
    UnknownCondition(String),
}

/// A syntax error in a decTest file.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParseError {
    /// The line number, from 1.
    pub line: usize,
    /// What is wrong.
    pub kind: ParseErrorKind,
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "line {}: ", self.line)?;
        match &self.kind {
            ParseErrorKind::UnclosedQuote => {
                write!(formatter, "a quoted token has no closing quote")
            }
            ParseErrorKind::UnknownDirective(keyword) => {
                write!(formatter, "unknown directive `{keyword}`")
            }
            ParseErrorKind::DirectiveValue(line) => write!(formatter, "invalid directive `{line}`"),
            ParseErrorKind::MissingDirective(keyword) => {
                write!(formatter, "a test comes before the `{keyword}` directive")
            }
            ParseErrorKind::Malformed => write!(formatter, "a test has no `->` or no result"),
            ParseErrorKind::Arity => write!(formatter, "a test has the wrong operand count"),
            ParseErrorKind::UnknownCondition(name) => {
                write!(formatter, "unknown condition `{name}`")
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// A token of a line, and whether it was quoted.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Token {
    value: String,
    quoted: bool,
}

impl Token {
    /// Returns whether the token is the unquoted `->` between the operands
    /// and the result.
    fn is_arrow(&self) -> bool {
        !self.quoted && self.value == "->"
    }
}

/// Splits a line into tokens, and drops a comment.
fn tokenize(line: &str) -> Result<Vec<Token>, ParseErrorKind> {
    let mut tokens = Vec::new();
    let mut rest = line.trim_start_matches([' ', '\t']);
    while !rest.is_empty() {
        let quote = rest
            .chars()
            .next()
            .filter(|first| ['\'', '"'].contains(first));
        let (token, after) = match quote {
            Some(quote) => quoted(&rest[1..], quote)?,
            None if rest.starts_with("--") => break,
            None => {
                let end = rest.find([' ', '\t']).unwrap_or(rest.len());
                let token = Token {
                    value: rest[..end].to_owned(),
                    quoted: false,
                };
                (token, &rest[end..])
            }
        };
        tokens.push(token);
        rest = after.trim_start_matches([' ', '\t']);
    }
    Ok(tokens)
}

/// Reads a quoted token after its opening quote. A doubled quote stands for
/// one quote character. Returns the token and the text after it.
fn quoted(text: &str, quote: char) -> Result<(Token, &str), ParseErrorKind> {
    let mut value = String::new();
    let mut characters = text.char_indices().peekable();
    while let Some((index, character)) = characters.next() {
        if character != quote {
            value.push(character);
            continue;
        }
        if characters.next_if(|&(_, next)| next == quote).is_some() {
            value.push(quote);
            continue;
        }
        let token = Token {
            value,
            quoted: true,
        };
        return Ok((token, &text[index + quote.len_utf8()..]));
    }
    Err(ParseErrorKind::UnclosedQuote)
}

/// The directive settings while a file is read. The first four have no
/// default.
struct Settings {
    precision: Option<u32>,
    rounding: Option<Rounding>,
    max_exponent: Option<i32>,
    min_exponent: Option<i32>,
    extended: bool,
    clamp: bool,
}

impl Settings {
    /// Applies a directive.
    fn apply(&mut self, keyword: &str, value: &str) -> Result<(), ParseErrorKind> {
        let invalid = || ParseErrorKind::DirectiveValue(format!("{keyword}: {value}"));
        let flag = || match value {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(invalid()),
        };
        match keyword.to_ascii_lowercase().as_str() {
            "precision" => self.precision = Some(value.parse().map_err(|_| invalid())?),
            "rounding" => self.rounding = Some(value.parse().map_err(|_| invalid())?),
            "maxexponent" => self.max_exponent = Some(value.parse().map_err(|_| invalid())?),
            "minexponent" => self.min_exponent = Some(value.parse().map_err(|_| invalid())?),
            "extended" => self.extended = flag()?,
            "clamp" => self.clamp = flag()?,
            "version" => {}
            _ => return Err(ParseErrorKind::UnknownDirective(keyword.to_owned())),
        }
        Ok(())
    }

    /// Returns the context of a test.
    fn context(&self) -> Result<Context, ParseErrorKind> {
        Ok(Context {
            precision: self
                .precision
                .ok_or(ParseErrorKind::MissingDirective("precision"))?,
            rounding: self
                .rounding
                .ok_or(ParseErrorKind::MissingDirective("rounding"))?,
            max_exponent: self
                .max_exponent
                .ok_or(ParseErrorKind::MissingDirective("maxexponent"))?,
            min_exponent: self
                .min_exponent
                .ok_or(ParseErrorKind::MissingDirective("minexponent"))?,
            extended: self.extended,
            clamp: self.clamp,
        })
    }
}

/// Parses the text of a decTest file.
///
/// # Errors
///
/// Returns a [`ParseError`] at the first line that is not a comment, a
/// directive, or a test.
pub fn parse(text: &str) -> Result<Vec<Test>, ParseError> {
    let mut settings = Settings {
        precision: None,
        rounding: None,
        max_exponent: None,
        min_exponent: None,
        extended: true,
        clamp: false,
    };
    let mut tests = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        let error = |kind| ParseError { line: number, kind };
        let tokens = tokenize(line.trim_end_matches('\r')).map_err(error)?;
        let Some(first) = tokens.first() else {
            continue;
        };
        if !first.quoted && first.value.ends_with(':') && tokens.len() == 2 {
            let keyword = first.value.trim_end_matches(':');
            settings.apply(keyword, &tokens[1].value).map_err(error)?;
            continue;
        }
        let context = settings.context().map_err(error)?;
        tests.push(parse_test(number, &tokens, context).map_err(error)?);
    }
    Ok(tests)
}

/// Parses the tokens of a test line.
fn parse_test(line: usize, tokens: &[Token], context: Context) -> Result<Test, ParseErrorKind> {
    let arrow = tokens
        .iter()
        .position(Token::is_arrow)
        .ok_or(ParseErrorKind::Malformed)?;
    if arrow < 2 || arrow + 1 >= tokens.len() {
        return Err(ParseErrorKind::Malformed);
    }
    let operation = Operation::from_name(&tokens[1].value);
    let operands: Vec<String> = tokens[2..arrow]
        .iter()
        .map(|token| token.value.clone())
        .collect();
    if operation
        .arity()
        .is_some_and(|arity| arity != operands.len())
    {
        return Err(ParseErrorKind::Arity);
    }
    let conditions = tokens[arrow + 2..]
        .iter()
        .try_fold(Status::NONE, |conditions, token| {
            Status::from_name(&token.value)
                .map(|condition| conditions | condition)
                .ok_or_else(|| ParseErrorKind::UnknownCondition(token.value.clone()))
        })?;
    Ok(Test {
        line,
        id: tokens[0].value.clone(),
        operation,
        operands,
        result: tokens[arrow + 1].value.clone(),
        conditions,
        context,
    })
}

/// Returns the paths of the decTest files whose names start with `prefix`,
/// for example `dd`, in name order.
///
/// # Panics
///
/// Panics when [`DIRECTORY`] cannot be read.
#[must_use]
pub fn files(prefix: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(DIRECTORY)
        .expect("build.rs extracts the decTest files")
        .map(|entry| entry.expect("the decTest directory can be read").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "decTest")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(prefix))
        })
        .collect();
    paths.sort();
    paths
}

/// Reads and parses the decTest file with a name, for example
/// `ddAdd.decTest`.
///
/// # Panics
///
/// Panics when the file cannot be read or parsed. The pinned files parse.
#[must_use]
pub fn read(name: &str) -> Vec<Test> {
    let path = PathBuf::from(DIRECTORY).join(name);
    let text = fs::read_to_string(&path).expect("build.rs extracts the decTest files");
    parse(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{Binary, Context, Operation, ParseErrorKind, Rounding, Status, Test, parse};

    const SAMPLE: &str = "\
-- a comment\r
version: 2.62\r
precision:   16\r
maxExponent: 384\r
minExponent: -383\r
extended:    1\r
clamp:       1\r
rounding:    half_even\r
\r
ddadd001 add 1       1       ->  2\r
ddadd011 add '0.4444444444444446'  '0.5555555555555555' -> '1.000000000000000' Inexact Rounded\r
ddcan202 add  0E+384 #77ffff3fcff3fcff        -> #77fcff3fcff3fcff  -- a comment\r
ddabs900 abs  # -> NaN Invalid_operation\r
ddtst001 toSci 'it''s' -> NaN conversion_SYNTAX\r
";

    #[test]
    fn parses_directives_tokens_and_conditions() {
        let tests = parse(SAMPLE).expect("the sample parses");
        assert_eq!(tests.len(), 5);
        let context = Context {
            precision: 16,
            rounding: Rounding::HalfEven,
            max_exponent: 384,
            min_exponent: -383,
            extended: true,
            clamp: true,
        };
        assert_eq!(
            tests[1],
            Test {
                line: 11,
                id: String::from("ddadd011"),
                operation: Operation::Binary(Binary::Add),
                operands: vec![
                    String::from("0.4444444444444446"),
                    String::from("0.5555555555555555")
                ],
                result: String::from("1.000000000000000"),
                conditions: Status::INEXACT | Status::ROUNDED,
                context,
            }
        );
        assert_eq!(tests[2].operands[1], "#77ffff3fcff3fcff");
        assert_eq!(tests[3].operands, [String::from("#")]);
        assert_eq!(tests[4].operation, Operation::ToSci);
        assert_eq!(tests[4].operands, [String::from("it's")]);
        assert_eq!(tests[4].conditions, Status::CONVERSION_SYNTAX);
    }

    #[test]
    fn rejects_malformed_lines() {
        let context = "precision: 16\nrounding: half_even\nmaxexponent: 384\nminexponent: -383\n";
        let kind = |line: &str| {
            parse(&format!("{context}{line}"))
                .expect_err("the line is invalid")
                .kind
        };
        assert_eq!(kind("x add '1 -> 2"), ParseErrorKind::UnclosedQuote);
        assert_eq!(kind("x add 1 1 2"), ParseErrorKind::Malformed);
        assert_eq!(kind("x add 1 ->"), ParseErrorKind::Malformed);
        assert_eq!(kind("x add 1 -> 2"), ParseErrorKind::Arity);
        assert_eq!(
            kind("x add 1 1 -> 2 Inexactly"),
            ParseErrorKind::UnknownCondition(String::from("Inexactly"))
        );
        assert_eq!(
            kind("dectest: ddAdd"),
            ParseErrorKind::UnknownDirective(String::from("dectest"))
        );
        assert_eq!(
            parse("x add 1 1 -> 2").expect_err("no context").kind,
            ParseErrorKind::MissingDirective("precision")
        );
    }
}
